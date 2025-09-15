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

def _storage_from_opt(opt):
    """
    Returns (storage_dtype_str, bytes_per_param).
    Priority:
      1) opt.Storage.param_bytes (2 or 4),
      2) opt.Storage.dtype ('float16'/'fp16'/'half' or 'float32'),
      3) default float32 (4 bytes).
    """
    dtype = "float32"
    bpp = 4
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

def _storage_conf(opt):
    """Returns a normalized storage config dict:
       {'mode': 'fp32'|'fp16'|'int8_tensor'|'int8_per_channel',
        'compress': 'none'|'npz'|'gzip',
        'percentile': float in (0,1] }
    """
    mode = "fp32"; compress = "none"; perc = 1.0
    if hasattr(opt, "Storage"):
        S = opt.Storage
        if hasattr(S, "mode") and S.mode:
            mode = str(S.mode).lower()
        if hasattr(S, "compress") and S.compress:
            compress = str(S.compress).lower()
        if hasattr(S, "percentile") and S.percentile:
            try:
                perc = float(S.percentile)
            except Exception:
                perc = 1.0
            perc = min(max(perc, 0.5), 1.0)
    return {"mode": mode, "compress": compress, "percentile": perc}

def _bytes_per_param_from_mode(mode: str) -> int:
    if mode in ("int8_tensor","int8_per_channel"): return 1
    if mode == "fp16": return 2
    return 4  # fp32 default

def _np_dtype_from_mode(mode: str):
    if mode == "fp16": return np.float16
    return np.float32

def _save_array(path_base: str, arr: np.ndarray, compress: str):
    """Save as raw .bin (none), or .npz (np.savez_compressed), or .gz."""
    base, ext = path_base, ""
    if compress == "npz":
        np.savez_compressed(base + ".npz", a=arr)
    elif compress == "gzip":
        import gzip
        with gzip.open(base + ".gz", "wb") as f:
            f.write(arr.tobytes(order="C"))
    else:
        with open(base, "wb") as f:
            f.write(arr.tobytes(order="C"))

def _load_array(path_base: str, dtype: np.dtype):
    """Load from .npz/.gz/raw, in that order of preference."""
    p_npz = path_base + ".npz"; p_gz = path_base + ".gz"; p_raw = path_base
    if os.path.exists(p_npz):
        z = np.load(p_npz); return z["a"].astype(dtype, copy=False)
    if os.path.exists(p_gz):
        import gzip
        with gzip.open(p_gz, "rb") as f:
            b = f.read()
        return np.frombuffer(b, dtype=dtype)
    with open(p_raw, "rb") as f:
        b = f.read()
    return np.frombuffer(b, dtype=dtype)

def _sym_quantize_tensor(x: np.ndarray, percentile: float = 1.0):
    """Per-tensor symmetric int8 quantization. Returns (q:int8, scale:float32)."""
    x = x.astype(np.float32, copy=False)
    if percentile < 1.0:
        max_abs = float(np.percentile(np.abs(x), 100.0*percentile))
    else:
        max_abs = float(np.max(np.abs(x))) if x.size else 1.0
    scale = max(max_abs / 127.0, 1e-12)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, np.float32(scale)

def _sym_quantize_per_outrow(W: np.ndarray, percentile: float = 1.0):
    """Per-output-channel symmetric quant for weight matrices (rows=outputs).
       Returns (q:int8 same shape, scales:float32 row vector)."""
    W = W.astype(np.float32, copy=False)
    if W.ndim != 2:
        raise ValueError("Expected 2D weight matrix")
    out = W.shape[0]
    q = np.empty_like(W, dtype=np.int8)
    scales = np.empty((out,), dtype=np.float32)
    for i in range(out):
        row = W[i]
        if percentile < 1.0:
            max_abs = float(np.percentile(np.abs(row), 100.0*percentile))
        else:
            max_abs = float(np.max(np.abs(row))) if row.size else 1.0
        s = max(max_abs / 127.0, 1e-12)
        scales[i] = s
        q[i] = np.clip(np.round(row / s), -127, 127).astype(np.int8)
    return q, scales

def _dequant(q: np.ndarray, scale):
    if isinstance(scale, np.ndarray):
        # per-row scales: broadcast
        return (q.astype(np.float32) * scale[:,None])
    else:
        return (q.astype(np.float32) * float(scale)).astype(np.float32)


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

def save_model(model:MLP, model_path:str, storage_conf=None):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if storage_conf is None:
        storage_conf = {"mode":"fp32","compress":"none","percentile":1.0}

    mode = storage_conf["mode"]
    compress = storage_conf["compress"]
    perc = storage_conf["percentile"]
    float_store = _np_dtype_from_mode(mode)

    for i in range(len(model.net)):
        layer = model.net[i]
        W = layer[0].weight.detach().cpu().numpy()
        B = layer[0].bias.detach().cpu().numpy()

        if mode == "fp32" or mode == "fp16":
            _save_array(os.path.join(model_path, f"{i}-W"), W.astype(float_store), compress)
            _save_array(os.path.join(model_path, f"{i}-B"), B.astype(float_store), compress)

        elif mode == "int8_tensor":
            qW, sW = _sym_quantize_tensor(W, percentile=perc)
            qB, sB = _sym_quantize_tensor(B, percentile=perc)
            _save_array(os.path.join(model_path, f"{i}-W.q8"), qW, compress)
            _save_array(os.path.join(model_path, f"{i}-B.q8"), qB, compress)
            _save_array(os.path.join(model_path, f"{i}-W.scale"), np.array([sW], dtype=np.float32), compress)
            _save_array(os.path.join(model_path, f"{i}-B.scale"), np.array([sB], dtype=np.float32), compress)

        elif mode == "int8_per_channel":
            if W.ndim != 2:
                raise ValueError("int8_per_channel expects 2D weight")
            qW, sW = _sym_quantize_per_outrow(W, percentile=perc)
            qB, sB = _sym_quantize_tensor(B, percentile=perc)
            _save_array(os.path.join(model_path, f"{i}-W.q8"), qW, compress)
            _save_array(os.path.join(model_path, f"{i}-W.scale"), sW, compress)
            _save_array(os.path.join(model_path, f"{i}-B.q8"), qB, compress)
            _save_array(os.path.join(model_path, f"{i}-B.scale"), np.array([sB], dtype=np.float32), compress)

        else:
            raise ValueError(f"Unknown Storage.mode: {mode}")


def load_model(model_path, hyper, storage_conf=None):
    model = MLP(**hyper)
    if storage_conf is None:
        storage_conf = {"mode":"fp32","compress":"none","percentile":1.0}

    mode = storage_conf["mode"]
    for i in range(len(model.net)):
        layer = model.net[i]
        W_shape = layer[0].weight.shape
        B_shape = layer[0].bias.shape

        if mode in ("fp32","fp16"):
            W = _load_array(os.path.join(model_path, f"{i}-W"), np.float32).reshape(W_shape)
            B = _load_array(os.path.join(model_path, f"{i}-B"), np.float32).reshape(B_shape)

        elif mode == "int8_tensor":
            qW = _load_array(os.path.join(model_path, f"{i}-W.q8"), np.int8).reshape(W_shape)
            qB = _load_array(os.path.join(model_path, f"{i}-B.q8"), np.int8).reshape(B_shape)
            sW = _load_array(os.path.join(model_path, f"{i}-W.scale"), np.float32).ravel()[0]
            sB = _load_array(os.path.join(model_path, f"{i}-B.scale"), np.float32).ravel()[0]
            W = _dequant(qW, sW); B = _dequant(qB, sB)

        elif mode == "int8_per_channel":
            qW = _load_array(os.path.join(model_path, f"{i}-W.q8"), np.int8).reshape(W_shape)
            sW = _load_array(os.path.join(model_path, f"{i}-W.scale"), np.float32)
            if sW.ndim != 1 or sW.shape[0] != W_shape[0]:
                raise ValueError("Bad per-channel scale shape")
            qB = _load_array(os.path.join(model_path, f"{i}-B.q8"), np.int8).reshape(B_shape)
            sB = _load_array(os.path.join(model_path, f"{i}-B.scale"), np.float32).ravel()[0]
            W = _dequant(qW, sW); B = _dequant(qB, sB)

        else:
            raise ValueError(f"Unknown Storage.mode: {mode}")

        with torch.no_grad():
            layer[0].weight.data = torch.tensor(W, dtype=torch.float32)
            layer[0].bias.data   = torch.tensor(B, dtype=torch.float32)

    return model



def load_model_from_files(model_path: str, in_dim: int, layer: int, act: str, output_act: bool, w0: int,
                          storage_conf=None):
    if storage_conf is None:
        storage_conf = {"mode":"fp32","compress":"none","percentile":1.0}
    mode = storage_conf["mode"]

    # Infer hidden/out from first and last layer files
    def _count_elems(path_base: str):
        for ext in (".npz",".gz",""):
            p = path_base + ext
            if os.path.exists(p):
                sz = os.path.getsize(p)
                if ext == ".npz":
                    # cannot infer elem count from zip; we’ll load to infer
                    a = _load_array(path_base, np.int8 if path_base.endswith(".q8") else np.float32)
                    return a.size
                else:
                    # raw stream: need bytes-per-elem by mode
                    bpe = 1 if path_base.endswith(".q8") or mode.startswith("int8") else (2 if mode=="fp16" else 4)
                    return sz // bpe
        raise FileNotFoundError(path_base)

    # first layer weight name base
    first_W_base = os.path.join(model_path, '0-W' + ('.q8' if mode.startswith('int8') else ''))
    nfloat = _count_elems(first_W_base)
    if in_dim <= 0 or (nfloat % in_dim) != 0:
        raise ValueError(f"First layer size mismatch: {nfloat} elems not divisible by in_dim={in_dim}")
    hidden = nfloat // in_dim

    # last layer infer out_dim from bias if present, else from weight
    b_last_base = os.path.join(model_path, f'{layer-1}-B' + ('.q8' if mode.startswith('int8') else ''))
    try:
        n_b = _count_elems(b_last_base)
        out_dim = n_b
    except Exception:
        last_W_base = os.path.join(model_path, f'{layer-1}-W' + ('.q8' if mode.startswith('int8') else ''))
        n_w = _count_elems(last_W_base)
        if n_w % hidden != 0:
            raise ValueError(f"Last layer size mismatch: {n_w} elems not divisible by hidden={hidden}")
        out_dim = n_w // hidden

    model = MLP(input=int(in_dim), output=int(out_dim), hidden=int(hidden),
                layer=int(layer), act=act, output_act=bool(output_act), w0=int(w0))

    # Load tensors via unified loader above
    loaded = load_model(model_path, model.hyper, storage_conf=storage_conf)
    return loaded, int(out_dim)


def save_tree_models(tree_mlp:OctTreeMLP, model_dir:str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    S = _storage_conf(tree_mlp.opt)
    # write all node models with chosen storage
    for node in tree_mlp.node_list:
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        save_model(model=node.net, model_path=model_path, storage_conf=S)

    # norm stats + opt as before
    stats = {k: _jsonable(v) for k, v in dict(tree_mlp.side_info).items()}
    try: shape = list(tree_mlp.data.shape)
    except Exception: shape = list(getattr(tree_mlp, "origin_shape", []))
    stats["shape"] = shape
    raw_dt = stats.get("raw_dtype", stats.get("dtype", "uint16"))
    stats["raw_dtype"] = str(np.dtype(raw_dt))
    try:
        itemsize = np.dtype(stats["raw_dtype"]).itemsize
        stats["origin_bytes"] = int((int(shape[0])*int(shape[1])*int(shape[2])*int(shape[3])) * itemsize)
    except Exception:
        stats["origin_bytes"] = None
    with open(os.path.join(model_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f)

    OmegaConf.save(tree_mlp.opt, os.path.join(model_dir, 'opt.yaml'))

    # storage meta
    meta = {"mode": S["mode"], "compress": S["compress"], "percentile": S["percentile"]}
    with open(os.path.join(model_dir, "storage_meta.json"), "w") as f:
        json.dump(meta, f)



def load_tree_models(model_dir:str):
    opt_path = os.path.join(model_dir, 'opt.yaml')
    opt = OmegaConf.load(opt_path)

    # storage meta
    meta_path = os.path.join(model_dir, "storage_meta.json")
    S = {"mode":"fp32","compress":"none","percentile":1.0}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            m = json.load(f)
        if isinstance(m, dict):
            S.update(m)

    # norm stats
    stats_path = os.path.join(model_dir, "norm_stats.json")
    origin_shape = None; saved_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            saved_stats = json.load(f)
        if isinstance(saved_stats.get("shape", None), list):
            origin_shape = tuple(saved_stats["shape"])

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
            storage_conf=S,
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


