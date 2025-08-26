import os
import torch
import numpy as np
import struct
import json
from omegaconf import OmegaConf
from utils.Network import MLP
from utils.OctTree import OctTreeMLP
#addd
def _jsonable(o):
    import numpy as _np
    if isinstance(o, (_np.integer, _np.floating, _np.bool_)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, _np.dtype):
        return str(o)
    return o
#adddd
def save_model(model:MLP, model_path:str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(len(model.net)):
        layer = model.net[i]

        weight = layer[0].weight.detach().cpu().numpy()
        weight = np.array(weight).reshape(-1)
        weight_path = os.path.join(model_path, f'{i}-W')
        with open(weight_path, 'wb') as data_file:
            data_file.write(struct.pack('f'*len(weight), *weight))
        
        bias = layer[0].bias.detach().cpu().numpy()
        bias_path = os.path.join(model_path, f'{i}-B')        
        with open(bias_path, 'wb') as data_file:
            data_file.write(struct.pack('f'*len(bias), *bias))

def load_model(model_path, hyper):
    model = MLP(**hyper)
    for i in range(len(model.net)):
        layer = model.net[i]

        weight_shape = layer[0].weight.shape
        weight_path = os.path.join(model_path, f'{i}-W')
        with open(weight_path, 'rb') as data_file:
            data = np.array(struct.unpack('f'*weight_shape[0]*weight_shape[1], data_file.read())).astype(np.float32)
            data = np.reshape(data, (weight_shape[0], weight_shape[1]))
        with torch.no_grad():
            model.net[i][0].weight.data = torch.tensor(data)

        bias_shape = layer[0].bias.shape
        bias_path = os.path.join(model_path, f'{i}-B') 
        with open(bias_path, 'rb') as data_file:
            data = np.array(struct.unpack('f'*bias_shape[0], data_file.read())).astype(np.float32)
        with torch.no_grad():
            model.net[i][0].bias.data = torch.tensor(data)
    return model

def load_model_from_files(model_path: str, in_dim: int, layer: int, act: str, output_act: bool, w0: int):
    # infer hidden from 0-W
    w0_path = os.path.join(model_path, '0-W')
    if not os.path.exists(w0_path):
        raise FileNotFoundError(f"Missing weight file: {w0_path}")
    nbytes = os.path.getsize(w0_path)
    nfloat = nbytes // 4
    if in_dim <= 0 or (nfloat % in_dim) != 0:
        raise ValueError(f"First layer size mismatch at {w0_path}: {nfloat} floats not divisible by in_dim={in_dim}")
    hidden = nfloat // in_dim

    # infer out_dim from last -B, or fallback to last -W
    b_last_path = os.path.join(model_path, f'{layer-1}-B')
    if os.path.exists(b_last_path):
        out_dim = os.path.getsize(b_last_path) // 4
    else:
        w_last_path = os.path.join(model_path, f'{layer-1}-W')
        if not os.path.exists(w_last_path):
            raise FileNotFoundError(f"Missing both {b_last_path} and {w_last_path}")
        nbytes_last = os.path.getsize(w_last_path)
        nfloat_last = nbytes_last // 4
        if hidden <= 0 or (nfloat_last % hidden) != 0:
            raise ValueError(f"Last layer size mismatch at {w_last_path}: {nfloat_last} floats not divisible by hidden={hidden}")
        out_dim = nfloat_last // hidden

    model = MLP(input=int(in_dim), output=int(out_dim), hidden=int(hidden),
                layer=int(layer), act=act, output_act=bool(output_act), w0=int(w0))

    # load tensors
    for i in range(len(model.net)):
        weight_shape = model.net[i][0].weight.shape
        with open(os.path.join(model_path, f'{i}-W'), 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(weight_shape[0], weight_shape[1])
        with torch.no_grad():
            model.net[i][0].weight.data = torch.tensor(data)

        bias_shape = model.net[i][0].bias.shape
        with open(os.path.join(model_path, f'{i}-B'), 'rb') as f:
            bias = np.frombuffer(f.read(), dtype=np.float32)
        with torch.no_grad():
            model.net[i][0].bias.data = torch.tensor(bias)

    return model, int(out_dim)

def save_tree_models(tree_mlp:OctTreeMLP, model_dir:str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for node in tree_mlp.node_list:
        model = node.net
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        save_model(model=model, model_path=model_path)
    #addd
    # persist normalization stats + data shape so decode can be no-reference
    stats = {k: _jsonable(v) for k, v in dict(tree_mlp.side_info).items()}
    if hasattr(tree_mlp, "data") and tree_mlp.data is not None:
        # tree_mlp.data is a torch tensor or np array; we only need the shape
        try:
            shape = list(tree_mlp.data.shape)
        except Exception:
            shape = list(getattr(tree_mlp, "origin_shape", []))
    else:
        shape = []

    stats["shape"] = shape

    # raw dtype of the *original* file (default to whatever side_info recorded)
    raw_dt = stats.get("raw_dtype", stats.get("dtype", "uint16"))
    # normalize to a string so JSON is stable
    raw_dt_str = str(np.dtype(raw_dt))

    stats["raw_dtype"] = raw_dt_str

    # compute origin_bytes from shape × raw dtype (so decode never needs the source TIFF)
    try:
        itemsize = _np.dtype(raw_dt_str).itemsize
        stats["origin_bytes"] = int((int(shape[0]) * int(shape[1]) * int(shape[2]) * int(shape[3])) * itemsize)
    except Exception:
        # safe fallback if shape is missing
        stats["origin_bytes"] = None
    with open(os.path.join(model_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f)
    #adddd
    opt_path = os.path.join(model_dir, 'opt.yaml')
    OmegaConf.save(tree_mlp.opt, opt_path)

def load_tree_models(model_dir:str):
    opt_path = os.path.join(model_dir, 'opt.yaml')
    opt = OmegaConf.load(opt_path)
    #tree_mlp = OctTreeMLP(opt)
    # for node in tree_mlp.node_list:
    #     hyper = node.net.hyper
    #     model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
    #     model = load_model(model_path=model_path, hyper=hyper)
    #     node.net = model
    #addd
    # try to restore saved shape/stats so we don't need the source TIFF
    stats_path = os.path.join(model_dir, "norm_stats.json")
    origin_shape = None
    saved_stats = None
    if os.path.exists(stats_path):
        import json, numpy as _np
        with open(stats_path, "r") as f:
            saved_stats = json.load(f)
        if isinstance(saved_stats.get("shape", None), list):
            origin_shape = tuple(saved_stats["shape"])

    # Build WITHOUT reading the original image
    tree_mlp = OctTreeMLP(opt, origin_shape=origin_shape, load_data=False)
    def _load_node_recursive(node, in_dim: int):
        # Use layer/act/output_act/w0 from placeholder hyper,
        # but infer hidden & output from the saved files on disk.
        h = node.net.hyper
        model_path = os.path.join(model_dir, f"{node.level}-{node.di}-{node.hi}-{node.wi}")
        model, out_dim = load_model_from_files(
            model_path=model_path,
            in_dim=int(in_dim),
            layer=int(h['layer']),
            act=h['act'],
            output_act=bool(h['output_act']),
            w0=int(h['w0']),
        )
        node.net = model
        for child in node.children:
            _load_node_recursive(child, out_dim)

    # Kick off from the root with the true coordinate input size from opt
    _load_node_recursive(tree_mlp.base_node, int(opt.Network.input))
    # Restore normalization stats so inv-normalization is correct
    if saved_stats:
        if "dtype" in saved_stats:
            import numpy as _np
            saved_stats["dtype"] = _np.dtype(saved_stats["dtype"])
        tree_mlp.side_info.update(saved_stats)
    #adddd
    return tree_mlp