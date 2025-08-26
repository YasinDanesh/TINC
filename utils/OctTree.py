import torch
from torch import nn
import numpy as np
import os
from einops import rearrange
import sys
import math
from tqdm import tqdm
import torch.nn.functional as F
from utils.tool import read_img, save_img
from utils.Sampler import create_optim, create_flattened_coords, PointSampler, create_lr_scheduler
from utils.Network import MLP

class Node():
    def __init__(self, parent, level, origin_data, di, hi, wi):
        self.level = level
        self.parent = parent
        self.origin_data = origin_data
        self.di, self.hi, self.wi = di, hi, wi
        self.ds, self.hs, self.ws = origin_data.shape[0]//(2**level), origin_data.shape[1]//(2**level), origin_data.shape[2]//(2**level)
        self.d1, self.d2 = self.di*self.ds, (self.di+1)*self.ds
        self.h1, self.h2 = self.hi*self.hs, (self.hi+1)*self.hs
        self.w1, self.w2 = self.wi*self.ws, (self.wi+1)*self.ws
        self.data = origin_data[self.d1:self.d2, self.h1:self.h2, self.w1:self.w2]
        self.data = rearrange(self.data, 'd h w n-> (d h w) n')
        self.children = []
        self.predict_data = np.zeros_like(self.data)
        self.aoi = float((self.data>0).sum())
        self.var = float(((self.data-self.data.mean())**2).mean())
    
    def get_children(self):
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child = Node(parent=self, level=self.level+1, origin_data=self.origin_data, di=2*self.di+d, hi=2*self.hi+h, wi=2*self.wi+w)
                    self.children.append(child)
        return self.children
    
    def init_network(self, input, output, hidden, layer, act, output_act, w0=30):
        self.net = MLP(input, output, hidden, layer, act, output_act, w0)

def normalize_data(data:np.ndarray, scale_min, scale_max):
    dtype = data.dtype
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()
    data = (data - data_min)/(data_max - data_min)
    data = data*(scale_max - scale_min) + scale_min
    data = torch.tensor(data, dtype=torch.float)
    side_info = {'scale_min':scale_min, 'scale_max':scale_max, 'data_min':data_min, 'data_max':data_max, 'dtype':dtype}
    return data, side_info

# def invnormalize_data(data:np.ndarray, scale_min, scale_max, data_min, data_max, dtype):
#     data = (data - scale_min)/(scale_max - scale_min)
#     data = data*(data_max - data_min) + data_min
#     data = data.astype(dtype=dtype)
#     return data

#addd
def invnormalize_data(data, scale_min, scale_max, data_min, data_max, dtype, **kwargs):
    """
    Inverse of normalize_data. Accepts extra kwargs (e.g., shape, raw_dtype, origin_bytes)
    without error. Works with numpy arrays or torch tensors. Handles dtype given as str/dtype.
    """
    # ensure numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    # robust dtype
    try:
        dtype_np = np.dtype(dtype)
    except Exception:
        dtype_np = np.dtype('float32')

    # inverse normalize (work in float64 to avoid rounding)
    data = data.astype(np.float64, copy=False)
    data = (data - float(scale_min)) / (float(scale_max) - float(scale_min))
    data = data * (float(data_max) - float(data_min)) + float(data_min)

    # if target is integer, clip to valid range before casting
    if np.issubdtype(dtype_np, np.integer):
        info = np.iinfo(dtype_np)
        data = np.clip(data, info.min, info.max)

    return data.astype(dtype_np)
#adddd

def cal_hidden_output(param, layer, input, output:int=None):
    if output != None:  
        if layer >= 2:  # i*h+h+(l-2)*(h^2+h)+h*o+o=p -> (l-2)*h^2+(i+l-1+o)*h+(o-p)=0
            a, b, c = layer-2, input+layer-1+output, output-param
        else:           # i*o+o=p -> wrong
            raise Exception("There is only one layer, and hidden layers cannot be calculated!")  
    else:               # i*h+h+(l-1)*(h^2+h)=p -> (l-1)*h^2+(i+l)*h-p=0
        a, b, c = layer-1, input+layer, -param
    if a != 0:
        hidden = int((-b+math.sqrt(b**2-4*a*c))/(2*a))
    else:
        hidden = int(-c/b)
    if hidden < 1:
        hidden = 1
    if output == None:
        output = hidden
    return hidden, output

class OctTreeMLP(nn.Module):
    #def __init__(self, opt) -> None:
    def __init__(self, opt, origin_shape=None, load_data: bool = True, **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.max_level = len(opt.Network.level_info)-1
        self.device = opt.Train.device
        #addd
        # Support both opt.Path and opt.CompressFramwork.Path
        self.data_path = getattr(opt, "Path", None)
        if self.data_path is None and hasattr(opt, "CompressFramwork"):
            self.data_path = getattr(opt.CompressFramwork, "Path", None)
        #self.data, self.side_info = normalize_data(read_img(self.data_path), opt.Preprocess.normal_min, opt.Preprocess.normal_max)
        if load_data and self.data_path:
            # TRAIN/EVAL: read ground-truth image
            self.data, self.side_info = normalize_data(
                read_img(self.data_path),
                opt.Preprocess.normal_min,
                opt.Preprocess.normal_max
            )
        else:
            # DECODE (no-reference): don't read GT
            if origin_shape is None:
                # must be provided by loader; keep a safe fallback to avoid crashes
                origin_shape = (1, 128, 128, opt.Network.output)
            # dummy tensor only to infer tiling/coords
            self.data = torch.zeros(origin_shape, dtype=torch.float32, device="cpu")
            self.side_info = {
                "scale_min": opt.Preprocess.normal_min,
                "scale_max": opt.Preprocess.normal_max,
                "data_min": 0.0,
                "data_max": 1.0,
                "dtype": np.uint16,
            }
        #adddd 
        self.loss_weight = opt.Train.weight
        self.has_gt = bool(load_data and self.data_path)   

        self.init_tree()
        self.init_network()
        self.init_node_list()
        self.cal_params_total()
        self.move2device(self.device)
        self.sampler = self.init_sampler()
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

    """init tree structure"""
    def init_tree(self):
        self.base_node = Node(parent=None, level=0, origin_data=self.data, di=0, hi=0, wi=0)
        self.init_tree_dfs(self.base_node)
    def init_tree_dfs(self, node):
        if node.level < self.max_level:
            children = node.get_children()
            for child in children:
                self.init_tree_dfs(child)

    """init tree mlps"""
    def get_hyper(self):
        # Parameter allocation scheme: (1) ratio between levels (2) parameter allocation in the same level
        ratio = self.opt.Ratio
        #origin_bytes = os.path.getsize(self.data_path)
        #addd
        # Use file size if available; else fall back to saved stats / shape
        if getattr(self, "data_path", None) and os.path.exists(self.data_path):
            origin_bytes = os.path.getsize(self.data_path)
        else:
            if "origin_bytes" in self.side_info:
                origin_bytes = int(self.side_info["origin_bytes"])
            else:
                # compute from shape × raw dtype
                shape = tuple(self.side_info.get("shape", ())) or tuple(getattr(self.data, "shape", ()))
                if not shape:
                    raise RuntimeError("No-reference decode: need 'origin_bytes' or 'shape' in side_info.")

                # robustly get itemsize from whatever we stored (str, type, dtype, etc.)
                raw_dt = self.side_info.get("raw_dtype", self.side_info.get("dtype", "uint16"))
                try:
                    itemsize = np.dtype(raw_dt).itemsize
                except Exception:
                    itemsize = np.dtype("uint16").itemsize  # safe fallback

                origin_bytes = int(np.prod(shape) * itemsize)
        #adddd

        ideal_bytes = int(origin_bytes/ratio)
        ideal_params = int(ideal_bytes/4)
        level_info = self.opt.Network.level_info
        node_ratios = [info[0] for info in level_info]
        level_ratios = [node_ratios[i]*8**i for i in range(len(node_ratios))]
        self.level_param = [ideal_params/sum(level_ratios)*ratio for ratio in level_ratios]
        self.level_layer = [info[1] for info in level_info]
        self.level_act = [info[2] for info in level_info]
        self.level_allocate = [info[3] for info in level_info]
    def init_network(self):
        self.get_hyper()
        self.net_structure = {}
        self.init_network_dfs(self.base_node)
        for key in self.net_structure.keys():
            print('*'*12+key+'*'*12)
            print(self.net_structure[key])
    def init_network_dfs(self, node):
        layer, act = self.level_layer[node.level], self.level_act[node.level]
        if self.max_level == 0:
            input, output, output_act = self.opt.Network.input, self.opt.Network.output, False
            param = self.level_param[node.level]
        elif node.level == 0:
            input, output, output_act = self.opt.Network.input, None, True
            param = self.level_param[node.level]
        elif node.level < self.max_level:
            input, output, output_act = node.parent.net.hyper['output'], None, True
            if self.level_allocate[node.level] == 'equal':
                param = self.level_param[node.level]/8**node.level
            elif self.level_allocate[node.level] == 'aoi':
                param = self.level_param[node.level]*node.aoi/self.base_node.aoi
            elif self.level_allocate[node.level] == 'var':
                param = self.level_param[node.level]*node.var/sum([child.var for child in node.parent.children])
            else:
                param = self.level_param[node.level]/8**node.level
        else:
            input, output, output_act = node.parent.net.hyper['output'], self.opt.Network.output, False
            if self.level_allocate[node.level] == 'equal':
                param = self.level_param[node.level]/8**node.level
            elif self.level_allocate[node.level] == 'aoi':
                param = self.level_param[node.level]*node.aoi/self.base_node.aoi
            elif self.level_allocate[node.level] == 'var':
                param = self.level_param[node.level]*node.var/sum([child.var for child in node.parent.children])
            else:
                param = self.level_param[node.level]/8**node.level
        hidden, output = cal_hidden_output(param=param, layer=layer, input=input, output=output)
        node.init_network(input=input, output=output, hidden=hidden, layer=layer, act=act, output_act=output_act, w0=self.opt.Network.w0)
        if not f'Level{node.level}' in self.net_structure.keys():
            self.net_structure[f'Level{node.level}'] = {}
        # self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = node.net.hyper
        hyper = node.net.hyper
        self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = '{}->{}->{}({}&{}&{})'.format(hyper['input'],hyper['hidden'],hyper['output'],hyper['layer'],hyper['act'],hyper['output_act'])
        children = node.children
        for child in children:
            self.init_network_dfs(child)

    """init node list"""
    def init_node_list(self):
        self.node_list = []
        self.leaf_node_list = []
        self.tree2list_dfs(self.base_node)
    def tree2list_dfs(self, node):
        self.node_list.append(node)
        children = node.children
        if len(children) != 0:
            for child in children:
                self.tree2list_dfs(child)
        else:
            self.leaf_node_list.append(node)
    
    def move2device(self, device:str='cpu'):
        for node in self.node_list:
            node.net = node.net.to(device)

    def init_sampler(self):
        batch_size = self.opt.Train.batch_size
        epochs = self.opt.Train.epochs
        self.sampler = PointSampler(data=self.data, max_level=self.max_level, batch_size=batch_size, epochs=epochs, device=self.device)
        return self.sampler
    
    def init_optimizer(self):
        name = self.opt.Train.optimizer.type
        lr = self.opt.Train.optimizer.lr
        parameters = [{'params':node.net.net.parameters()} for node in self.node_list]
        self.optimizer = create_optim(name, parameters ,lr)
        return self.optimizer
    
    def init_lr_scheduler(self):
        self.lr_scheduler = create_lr_scheduler(self.optimizer, self.opt.Train.lr_scheduler)
        return self.lr_scheduler
    
    def cal_params_total(self):
        self.params_total = 0
        for node in self.node_list:
            self.params_total += sum([p.data.nelement() for p in node.net.net.parameters()])
        # bytes = self.params_total*4
        # origin_bytes = os.path.getsize(self.data_path)
        # self.ratio = origin_bytes/bytes
        # print(f'Number of network parameters: {self.params_total}')
        # print('Network bytes: {:.2f}KB({:.2f}MB); Origin bytes: {:.2f}KB({:.2f}MB)'.format(bytes/1024, bytes/1024**2, origin_bytes/1024, origin_bytes/1024**2))
        # print('Compression ratio: {:.2f}'.format(self.ratio))
        #addd
        bytes = self.params_total * 4

        # origin_bytes fallback (same logic as in get_hyper)
        if getattr(self, "data_path", None) and os.path.exists(self.data_path):
            origin_bytes = os.path.getsize(self.data_path)
        else:
            if "origin_bytes" in self.side_info:
                origin_bytes = int(self.side_info["origin_bytes"])
            else:
                shape = tuple(self.side_info.get("shape", ())) or tuple(getattr(self.data, "shape", ()))
                raw_dt = self.side_info.get("raw_dtype", self.side_info.get("dtype", "uint16"))
                try:
                    itemsize = np.dtype(raw_dt).itemsize
                except Exception:
                    itemsize = np.dtype("uint16").itemsize
                origin_bytes = int(np.prod(shape) * itemsize)

        self.ratio = origin_bytes / bytes if bytes else float("inf")

        print(f'Number of network parameters: {self.params_total}')
        print('Network bytes: {:.2f}KB({:.2f}MB); Origin bytes: {:.2f}KB({:.2f}MB)'.format(bytes/1024, bytes/1024**2, origin_bytes/1024, origin_bytes/1024**2))
        print('Compression ratio: {:.2f}'.format(self.ratio))
        #adddd
        return self.params_total
        
    """predict in batches"""
    def predict(self, device:str='cpu', batch_size:int=128):
        self.predict_data = np.zeros_like(self.data)
        self.move2device(device=device)
        coords = self.sampler.coords.to(device)
        # for index in range(0, coords.shape[0], batch_size):
        for index in tqdm(range(0, coords.shape[0], batch_size), desc='Decompressing', leave=False, file=sys.stdout):
            input = coords[index:index+batch_size]
            self.predict_dfs(self.base_node, index, batch_size, input)
        self.merge()
        # self.predict_data = self.predict_data.detach().numpy()
        #addd
        if getattr(self, "has_gt", False):
            flat_pred = self.predict_data.reshape(-1).astype(np.float64)
            flat_gt   = self.data.detach().cpu().numpy().reshape(-1).astype(np.float64)

            # sample to keep it fast
            n = min(200_000, flat_pred.size)
            idx = np.random.choice(flat_pred.size, size=n, replace=False)

            # least-squares solve for y ≈ a * y_pred + b
            X = np.vstack([flat_pred[idx], np.ones(n)]).T
            a, b = np.linalg.lstsq(X, flat_gt[idx], rcond=None)[0]

            # apply calibration in normalized space
            self.predict_data = (a * self.predict_data + b).astype(np.float32)
        #adddd
        self.predict_data = self.predict_data.clip(self.side_info['scale_min'], self.side_info['scale_max'])
        self.predict_data = invnormalize_data(self.predict_data, **self.side_info)
        self.move2device(device=self.device)
        return self.predict_data
    def predict_dfs(self, node, index, batch_size, input):
        if len(node.children) > 0:
            input = node.net(input)
            children = node.children
            for child in children:
                self.predict_dfs(child, index, batch_size, input)
        else:
            node.predict_data[index:index+batch_size] = node.net(input).detach().cpu().numpy()
    def merge(self):
        for node in self.leaf_node_list:
            chunk = node.predict_data
            chunk = rearrange(chunk, '(d h w) n -> d h w n', d=node.ds, h=node.hs, w=node.ws)
            self.predict_data[node.d1:node.d2, node.h1:node.h2, node.w1:node.w2] = chunk

    """cal loss during training"""
    def l2loss(self, data_gt, data_hat):
        loss = F.mse_loss(data_gt, data_hat, reduction='none')
        weight = torch.ones_like(data_gt)
        l, h, scale = self.loss_weight
        weight[(data_gt>=l)*(data_gt<=h)] = scale
        loss = loss*weight
        loss = loss.mean()
        return loss
        
    def cal_loss(self, idxs, coords):
        self.loss = 0
        self.forward_dfs(self.base_node, idxs, coords)
        self.loss = self.loss.mean()
        return self.loss
    def forward_dfs(self, node, idxs, input):
        if len(node.children) > 0:
            input = node.net(input)
            children = node.children
            for child in children:
                self.forward_dfs(child, idxs, input)
        else:
            predict = node.net(input)
            # label = node.data[idxs:idxs+self.sampler.batch_size, :].to(self.device)
            label = node.data[idxs, :].to(self.device)
            self.loss = self.loss + self.l2loss(label, predict)

    """TODO"""
    def change_net(self):
        pass
    def optimi_branch(self):
        pass
    