import os
import torch
import numpy as np
import struct
import json
from omegaconf import OmegaConf
from utils.Network import MLP
from utils.OctTree import OctTreeMLP
#addd
from utils.OctTree import _fit_affine, _tile_boxes

def _jsonable(o):
    import numpy as _np
    if isinstance(o, (_np.integer, _np.floating, _np.bool_)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, _np.dtype):
        return str(o)
    return o

# === storage helpers ===
def _storage_from_opt(opt):
    """
    Returns (storage_dtype_str, bytes_per_param).
    Priority:
      1) opt.Storage.param_bytes (2 or 4),
      2) opt.Storage.dtype ('float16'/'fp16'/'half' or 'float32'),
      3) default float32 (4 bytes).
    """
    dtype = "float16"
    bpp = 2
    try:
        if hasattr(opt, "Storage"):
            S = opt.Storage
            if hasattr(S, "param_bytes") and S.param_bytes in (2, 4):
                bpp = int(S.param_bytes)
                dtype = "float16" if bpp == 2 else "float32"
            elif hasattr(S, "dtype") and str(S.dtype).lower() in ("float16", "fp16", "half"):
                dtype, bpp = "float16", 2
            elif hasattr(S, "dtype") and str(S.dtype).lower() in ("float32", "fp32"):
                dtype, bpp = "float32", 4
    except Exception:
        pass
    return dtype, bpp

def _np_dtype_from_storage_str(s: str):
    s = str(s).lower()
    if s in ("float16", "fp16", "half"):
        return np.float16
    return np.float32
# === end helpers ===


def write_calibration(tree_mlp: OctTreeMLP, model_dir: str, pred_norm=None, gt_norm=None):
    """
    Write calib_global.json and (if 3+ levels) calib_leaves.npz using
    a precomputed normalized prediction and ground-truth.
    Never calls predict(); returns silently if data is missing.
    """
    try:
        # prefer explicitly provided arrays
        if pred_norm is None:
            pred_norm = getattr(tree_mlp, "_last_pred_norm", None)
        if gt_norm is None:
            gt_norm = getattr(tree_mlp, "_last_gt_norm", None)

        if pred_norm is None or gt_norm is None:
            return  # nothing to do; do NOT decode here

        # global (a,b)
        a, b = _fit_affine(pred_norm, gt_norm, sample=min(200_000, pred_norm.size))
        with open(os.path.join(model_dir, "calib_global.json"), "w") as f:
            json.dump({"a": float(a), "b": float(b)}, f)

        # per-leaf for 3+ levels
        num_levels = len(tree_mlp.opt.Network.level_info)
        if num_levels >= 3:
            boxes = _tile_boxes(pred_norm.shape, num_levels=num_levels)
            d = 2 ** (num_levels - 1)
            A = np.zeros((d, d, d), dtype=np.float32)
            B = np.zeros((d, d, d), dtype=np.float32)
            idx = 0
            for zi in range(d):
                for yi in range(d):
                    for xi in range(d):
                        z0,z1,y0,y1,x0,x1 = boxes[idx]; idx += 1
                        p = pred_norm[z0:z1, y0:y1, x0:x1, 0]
                        g = gt_norm  [z0:z1, y0:y1, x0:x1, 0]
                        a_, b_ = _fit_affine(p, g, sample=None)
                        A[zi,yi,xi] = a_
                        B[zi,yi,xi] = b_
            np.savez_compressed(os.path.join(model_dir, "calib_leaves.npz"), a=A, b=B)
    except Exception:
        # don't fail the save on calibration problems
        pass

def write_residuals(tree_mlp: OctTreeMLP, model_dir: str, pred_norm=None, gt_norm=None):
    """
    Write residual_leaves.npz with per-leaf MSE in the NORMALIZED space.
    Never calls predict(); uses cached arrays produced by OctTreeMLP.predict().
    """
    try:
        if pred_norm is None:
            pred_norm = getattr(tree_mlp, "_last_pred_norm", None)
        if gt_norm is None:
            gt_norm = getattr(tree_mlp, "_last_gt_norm", None)
        if pred_norm is None or gt_norm is None:
            return  # nothing to do without cached arrays

        num_levels = len(tree_mlp.opt.Network.level_info)
        boxes = _tile_boxes(pred_norm.shape, num_levels=num_levels)
        d = 2 ** (num_levels - 1)  # leaves per axis

        MSE = np.zeros((d, d, d), dtype=np.float32)
        idx = 0
        for zi in range(d):
            for yi in range(d):
                for xi in range(d):
                    z0,z1,y0,y1,x0,x1 = boxes[idx]; idx += 1
                    p = pred_norm[z0:z1, y0:y1, x0:x1, 0].astype(np.float64, copy=False)
                    g = gt_norm  [z0:z1, y0:y1, x0:x1, 0].astype(np.float64, copy=False)
                    MSE[zi, yi, xi] = float(np.mean((p - g)**2))

        np.savez_compressed(os.path.join(model_dir, "residual_leaves.npz"), mse=MSE)
    except Exception:
        # don't break saving if residual export fails
        pass

def save_model(model:MLP, model_path:str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(len(model.net)):
        layer = model.net[i]
        
        weight = layer[0].weight.detach().cpu().numpy()
        bias   = layer[0].bias.detach().cpu().numpy()
        weight_path = os.path.join(model_path, f'{i}-W')
        bias_path   = os.path.join(model_path, f'{i}-B')

        # Default to float32; caller (save_tree_models) will reopen and rewrite as desired dtype if needed
        with open(weight_path, 'wb') as fW:
            fW.write(np.asarray(weight, dtype=np.float32).tobytes(order='C'))
        with open(bias_path, 'wb') as fB:
            fB.write(np.asarray(bias, dtype=np.float32).tobytes(order='C'))


def load_model(model_path, hyper, storage_dtype_str: str = "float32"):
    model = MLP(**hyper)
    storage_np_dtype = _np_dtype_from_storage_str(storage_dtype_str)

    for i in range(len(model.net)):
        layer = model.net[i]

        weight_shape = layer[0].weight.shape
        weight_path  = os.path.join(model_path, f'{i}-W')
        with open(weight_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=storage_np_dtype)
        data = data.reshape(weight_shape)
        with torch.no_grad():
            model.net[i][0].weight.data = torch.tensor(data, dtype=torch.float32)

        bias_shape = layer[0].bias.shape
        bias_path  = os.path.join(model_path, f'{i}-B')
        with open(bias_path, 'rb') as f:
            b = np.frombuffer(f.read(), dtype=storage_np_dtype)
        with torch.no_grad():
            model.net[i][0].bias.data = torch.tensor(b, dtype=torch.float32)
    return model


def load_model_from_files(model_path: str, in_dim: int, layer: int,
                          act: str, output_act: bool, w0: int,
                          storage_bytes: int = 4, storage_dtype: str = "float32"):
    # infer hidden from 0-W
    w0_path = os.path.join(model_path, '0-W')
    if not os.path.exists(w0_path):
        raise FileNotFoundError(f"Missing weight file: {w0_path}")
    nbytes = os.path.getsize(w0_path)
    if storage_bytes not in (2, 4):
        storage_bytes = 4
    nfloat = nbytes // storage_bytes
    if in_dim <= 0 or (nfloat % in_dim) != 0:
        raise ValueError(f"First layer size mismatch at {w0_path}: {nfloat} elems not divisible by in_dim={in_dim}")
    hidden = nfloat // in_dim

    # infer out_dim from last -B, or fallback to last -W
    b_last_path = os.path.join(model_path, f'{layer-1}-B')
    if os.path.exists(b_last_path):
        out_dim = os.path.getsize(b_last_path) // storage_bytes
    else:
        w_last_path = os.path.join(model_path, f'{layer-1}-W')
        if not os.path.exists(w_last_path):
            raise FileNotFoundError(f"Missing both {b_last_path} and {w_last_path}")
        nbytes_last = os.path.getsize(w_last_path)
        nfloat_last = nbytes_last // storage_bytes
        if hidden <= 0 or (nfloat_last % hidden) != 0:
            raise ValueError(f"Last layer size mismatch at {w_last_path}: {nfloat_last} elems not divisible by hidden={hidden}")
        out_dim = nfloat_last // hidden

    model = MLP(input=int(in_dim), output=int(out_dim), hidden=int(hidden),
                layer=int(layer), act=act, output_act=bool(output_act), w0=int(w0))

    # load tensors
    storage_np_dtype = _np_dtype_from_storage_str(storage_dtype)
    for i in range(len(model.net)):
        weight_shape = model.net[i][0].weight.shape
        with open(os.path.join(model_path, f'{i}-W'), 'rb') as f:
            data = np.frombuffer(f.read(), dtype=storage_np_dtype)
        data = data.reshape(weight_shape)
        with torch.no_grad():
            model.net[i][0].weight.data = torch.tensor(data, dtype=torch.float32)

        bias_shape = model.net[i][0].bias.shape
        with open(os.path.join(model_path, f'{i}-B'), 'rb') as f:
            bias = np.frombuffer(f.read(), dtype=storage_np_dtype)
        with torch.no_grad():
            model.net[i][0].bias.data = torch.tensor(bias, dtype=torch.float32)

    return model, int(out_dim)

def save_tree_models(tree_mlp:OctTreeMLP, model_dir:str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # decide storage (dtype + bytes/param)
    storage_dtype_str, bpp = _storage_from_opt(tree_mlp.opt)
    storage_np_dtype = _np_dtype_from_storage_str(storage_dtype_str)

    # 1) write all node models as float32 first (via save_model), then recast files to storage_np_dtype
    for node in tree_mlp.node_list:
        model = node.net
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        # write float32 files
        save_model(model=model, model_path=model_path)
        # recast files to desired dtype
        for i in range(len(model.net)):
            # rewrite W
            w_path = os.path.join(model_path, f'{i}-W')
            with open(w_path, 'rb') as f:
                w = np.frombuffer(f.read(), dtype=np.float32)
            with open(w_path, 'wb') as f:
                f.write(w.astype(storage_np_dtype).tobytes(order='C'))
            # rewrite B
            b_path = os.path.join(model_path, f'{i}-B')
            with open(b_path, 'rb') as f:
                b = np.frombuffer(f.read(), dtype=np.float32)
            with open(b_path, 'wb') as f:
                f.write(b.astype(storage_np_dtype).tobytes(order='C'))

    # --- persist normalization stats + shape (unchanged) ---
    stats = {k: _jsonable(v) for k, v in dict(tree_mlp.side_info).items()}
    if hasattr(tree_mlp, "data") and tree_mlp.data is not None:
        try:
            shape = list(tree_mlp.data.shape)
        except Exception:
            shape = list(getattr(tree_mlp, "origin_shape", []))
    else:
        shape = []
    stats["shape"] = shape
    raw_dt = stats.get("raw_dtype", stats.get("dtype", "uint16"))
    raw_dt_str = str(np.dtype(raw_dt))
    stats["raw_dtype"] = raw_dt_str
    try:
        itemsize = np.dtype(raw_dt_str).itemsize
        stats["origin_bytes"] = int((int(shape[0]) * int(shape[1]) * int(shape[2]) * int(shape[3])) * itemsize)
    except Exception:
        stats["origin_bytes"] = None
    with open(os.path.join(model_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f)

    # save opt
    opt_path = os.path.join(model_dir, 'opt.yaml')
    OmegaConf.save(tree_mlp.opt, opt_path)

    # --- NEW: write storage metadata so loader knows bpp/dtype ---
    meta = {"param_bytes": int(bpp), "storage_dtype": storage_dtype_str}
    with open(os.path.join(model_dir, "storage_meta.json"), "w") as f:
        json.dump(meta, f)

    # (optional) residuals export stays as-is; if you keep it, it doesn’t depend on param dtype
    # ... (leave your residual export block unchanged or remove if not needed)


def load_tree_models(model_dir:str):
    opt_path = os.path.join(model_dir, 'opt.yaml')
    opt = OmegaConf.load(opt_path)

    # read storage meta (default to float32/4 if missing)
    meta_path = os.path.join(model_dir, "storage_meta.json")
    storage_dtype_str, bpp = "float16", 2
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            _m = json.load(f)
        if isinstance(_m, dict):
            if "storage_dtype" in _m:
                storage_dtype_str = str(_m["storage_dtype"])
            if "param_bytes" in _m and _m["param_bytes"] in (2, 4):
                bpp = int(_m["param_bytes"])

    # try to restore saved shape/stats so we don't need the source TIFF
    stats_path = os.path.join(model_dir, "norm_stats.json")
    origin_shape = None
    saved_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            saved_stats = json.load(f)
        if isinstance(saved_stats.get("shape", None), list):
            origin_shape = tuple(saved_stats["shape"])

    # Build WITHOUT reading the original image
    tree_mlp = OctTreeMLP(opt, origin_shape=origin_shape, load_data=False)
    tree_mlp.model_dir = model_dir

    def _load_node_recursive(node, in_dim: int):
        h = node.net.hyper
        model_path = os.path.join(model_dir, f"{node.level}-{node.di}-{node.hi}-{node.wi}")
        model, out_dim = load_model_from_files(
            model_path=model_path,
            in_dim=int(in_dim),
            layer=int(h['layer']),
            act=h['act'],
            output_act=bool(h['output_act']),
            w0=int(h['w0']),
            storage_bytes=int(bpp),
            storage_dtype=storage_dtype_str,
        )
        node.net = model
        for child in node.children:
            _load_node_recursive(child, out_dim)

    _load_node_recursive(tree_mlp.base_node, int(opt.Network.input))

    if saved_stats:
        if "dtype" in saved_stats:
            saved_stats["dtype"] = np.dtype(saved_stats["dtype"])
        tree_mlp.side_info.update(saved_stats)

    return tree_mlp

