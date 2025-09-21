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

def _storage_conf(opt):
    """
    Returns normalized storage config dict:
       {'mode': 'fp32'|'fp16'|'int8_tensor'|'int8_per_channel',
        'compress': 'none'|'npz',
        'percentile': float in (0,1] }
    """
    mode = "int8_tensor"; compress = "none"; perc = 0.995
    if hasattr(opt, "Storage"):
        S = opt.Storage
        if getattr(S, "mode", None):
            mode = str(S.mode).lower()
        if getattr(S, "compress", None):
            compress = "npz" if str(S.compress).lower() == "npz" else "none"
        if getattr(S, "percentile", None):
            try:
                perc = float(S.percentile)
            except Exception:
                perc = 0.995
            perc = min(max(perc, 0.5), 1.0)
    return {"mode": mode, "compress": compress, "percentile": perc}


def _np_dtype_from_mode(mode: str):
    if mode == "fp16": return np.float16
    return np.float32

def _save_array(path_base: str, arr: np.ndarray, compress: str):
    with open(path_base, "wb") as f:
        f.write(arr.tobytes(order="C"))

def _load_array(path_base: str, dtype: np.dtype):
    with open(path_base, "rb") as f:
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

def _pack_tensors_to_blob(named_arrays):
    """
    named_arrays: list of (name:str, arr:np.ndarray with final dtype/shape)
    returns (blob_u8: np.uint8, meta: dict[name -> {offset,nbytes,shape,dtype}])
    """
    offset = 0
    pieces = []
    meta = {}
    for name, arr in named_arrays:
        arr_c = np.ascontiguousarray(arr)
        b = arr_c.tobytes(order="C")
        nbytes = len(b)
        meta[name] = {
            "offset": int(offset),
            "nbytes": int(nbytes),
            "shape": list(arr_c.shape),
            "dtype": np.dtype(arr_c.dtype).str,
        }
        pieces.append(b)
        offset += nbytes
    blob = np.frombuffer(b"".join(pieces), dtype=np.uint8)
    return blob, meta

def _unpack_tensors_from_blob(blob_u8, meta):
    """
    blob_u8: np.uint8 array; meta: dict as above
    returns dict[name -> np.ndarray]
    """
    out = {}
    raw = memoryview(blob_u8.tobytes())  # contiguous bytes
    for name, info in meta.items():
        off = int(info["offset"]); nbytes = int(info["nbytes"])
        dt = np.dtype(info["dtype"]); shp = tuple(info["shape"])
        chunk = raw[off:off+nbytes]
        arr = np.frombuffer(chunk, dtype=dt).reshape(shp)
        out[name] = np.array(arr, copy=True)  # materialize
    return out


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

def save_model(model: MLP, model_path: str, storage_conf=None):
    """
    Save a single node's model as raw binary files only (compress: 'none').
    Tree-level compression (compress: 'npz') is handled by save_tree_models().
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if storage_conf is None:
        storage_conf = {"mode": "int8_tensor", "compress": "none", "percentile": 0.995}

    mode = storage_conf.get("mode", "int8_tensor")
    compress = storage_conf.get("compress", "none")
    perc = float(storage_conf.get("percentile", 0.995))

    if compress != "none":
        raise ValueError("save_model only supports compress='none'. "
                         "Use save_tree_models(model_dir) for compress='npz'.")

    float_store = _np_dtype_from_mode(mode)

    for i in range(len(model.net)):
        layer = model.net[i]
        W = layer[0].weight.detach().cpu().numpy()
        B = layer[0].bias.detach().cpu().numpy()

        if mode in ("fp32", "fp16"):
            _save_array(os.path.join(model_path, f"{i}-W"), W.astype(float_store), "none")
            _save_array(os.path.join(model_path, f"{i}-B"), B.astype(float_store), "none")

        elif mode == "int8_tensor":
            qW, sW = _sym_quantize_tensor(W, percentile=perc)
            qB, sB = _sym_quantize_tensor(B, percentile=perc)
            _save_array(os.path.join(model_path, f"{i}-W.q8"), qW, "none")
            _save_array(os.path.join(model_path, f"{i}-B.q8"), qB, "none")
            _save_array(os.path.join(model_path, f"{i}-W.scale"), np.array([sW], dtype=np.float32), "none")
            _save_array(os.path.join(model_path, f"{i}-B.scale"), np.array([sB], dtype=np.float32), "none")

        elif mode == "int8_per_channel":
            if W.ndim != 2:
                raise ValueError("int8_per_channel expects 2D weight")
            qW, sW = _sym_quantize_per_outrow(W, percentile=perc)
            qB, sB = _sym_quantize_tensor(B, percentile=perc)
            _save_array(os.path.join(model_path, f"{i}-W.q8"), qW, "none")
            _save_array(os.path.join(model_path, f"{i}-W.scale"), sW.astype(np.float32), "none")
            _save_array(os.path.join(model_path, f"{i}-B.q8"), qB, "none")
            _save_array(os.path.join(model_path, f"{i}-B.scale"), np.array([sB], dtype=np.float32), "none")

        else:
            raise ValueError(f"Unknown Storage.mode: {mode}")



def load_model(model_path, hyper, storage_conf=None):
    model = MLP(**hyper)
    if storage_conf is None:
        storage_conf = {"mode": "int8_tensor", "compress": "none", "percentile": 0.995}

    mode = storage_conf.get("mode", "int8_tensor")
    compress = storage_conf.get("compress", "none")

    if compress != "none":
        raise ValueError("load_model only supports compress='none'. Use load_tree_models(model_dir) for compress='npz'.")

    for i in range(len(model.net)):
        layer = model.net[i]
        W_shape = layer[0].weight.shape
        B_shape = layer[0].bias.shape

        if mode in ("fp32", "fp16"):
            dt = _np_dtype_from_mode(mode)   # np.float32 or np.float16
            W = _load_array(os.path.join(model_path, f"{i}-W"), dt).reshape(W_shape).astype(np.float32, copy=False)
            B = _load_array(os.path.join(model_path, f"{i}-B"), dt).reshape(B_shape).astype(np.float32, copy=False)

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
    """
    Infer hidden/out dims and load a node from raw files only (compress: 'none').
    Tree-level compressed loading is handled by load_tree_models().
    """
    if storage_conf is None:
        storage_conf = {"mode": "int8_tensor", "compress": "none", "percentile": 0.995}
    mode = storage_conf.get("mode", "int8_tensor")
    compress = storage_conf.get("compress", "none")

    if compress != "none":
        raise ValueError("load_model_from_files only supports compress='none'. "
                         "Use load_tree_models() for compress='npz'.")

    def _count_elems_raw(path_base: str):
        p = path_base
        if os.path.exists(p):
            sz = os.path.getsize(p)
            bpe = 1 if (path_base.endswith(".q8") or mode.startswith("int8")) else (2 if mode == "fp16" else 4)
            return sz // bpe
        raise FileNotFoundError(p)

    # Infer hidden from first layer W
    first_W_base = os.path.join(model_path, '0-W' + ('.q8' if mode.startswith('int8') else ''))
    nfloat = _count_elems_raw(first_W_base)
    if in_dim <= 0 or (nfloat % in_dim) != 0:
        raise ValueError(f"First layer size mismatch: {nfloat} elems not divisible by in_dim={in_dim}")
    hidden = nfloat // in_dim

    # Infer out from last bias if present; else from last W rows
    b_last_base = os.path.join(model_path, f'{layer-1}-B' + ('.q8' if mode.startswith('int8') else ''))
    try:
        n_b = _count_elems_raw(b_last_base)
        out_dim = n_b
    except Exception:
        last_W_base = os.path.join(model_path, f'{layer-1}-W' + ('.q8' if mode.startswith('int8') else ''))
        n_w = _count_elems_raw(last_W_base)
        if n_w % hidden != 0:
            raise ValueError(f"Last layer size mismatch: {n_w} elems not divisible by hidden={hidden}")
        out_dim = n_w // hidden

    model = MLP(input=int(in_dim), output=int(out_dim), hidden=int(hidden),
                layer=int(layer), act=act, output_act=bool(output_act), w0=int(w0))
    loaded = load_model(model_path, model.hyper, storage_conf=storage_conf)
    return loaded, int(out_dim)

def save_tree_models(tree_mlp:OctTreeMLP, model_dir:str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    S = _storage_conf(tree_mlp.opt)
    mode = S["mode"]; compress = S["compress"]; perc = S["percentile"]

    # Always write stats/opt/meta
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

    meta = {"mode": mode, "compress": compress, "percentile": perc, "format_version": 1}
    with open(os.path.join(model_dir, "storage_meta.json"), "w") as f:
        json.dump(meta, f)

    if compress == "npz":
        # Build a single list of all tensors across the entire tree
        named = []
        for node in tree_mlp.node_list:
            node_id = f"{node.level}-{node.di}-{node.hi}-{node.wi}"
            for li in range(len(node.net.net)):
                layer = node.net.net[li]
                W = layer[0].weight.detach().cpu().numpy()
                B = layer[0].bias.detach().cpu().numpy()

                if mode in ("fp32","fp16"):
                    named.append((f"{node_id}|{li}-W", W.astype(_np_dtype_from_mode(mode))))
                    named.append((f"{node_id}|{li}-B", B.astype(_np_dtype_from_mode(mode))))
                elif mode == "int8_tensor":
                    qW, sW = _sym_quantize_tensor(W, percentile=perc)
                    qB, sB = _sym_quantize_tensor(B, percentile=perc)
                    named.append((f"{node_id}|{li}-W.q8", qW))
                    named.append((f"{node_id}|{li}-B.q8", qB))
                    named.append((f"{node_id}|{li}-W.scale", np.array([sW], dtype=np.float32)))
                    named.append((f"{node_id}|{li}-B.scale", np.array([sB], dtype=np.float32)))
                elif mode == "int8_per_channel":
                    if W.ndim != 2:
                        raise ValueError("int8_per_channel expects 2D weight")
                    qW, sW = _sym_quantize_per_outrow(W, percentile=perc)
                    qB, sB = _sym_quantize_tensor(B, percentile=perc)
                    named.append((f"{node_id}|{li}-W.q8", qW))
                    named.append((f"{node_id}|{li}-W.scale", sW.astype(np.float32)))
                    named.append((f"{node_id}|{li}-B.q8", qB))
                    named.append((f"{node_id}|{li}-B.scale", np.array([sB], dtype=np.float32)))
                else:
                    raise ValueError(f"Unknown Storage.mode: {mode}")

        blob, meta_tbl = _pack_tensors_to_blob(named)
        np.savez_compressed(os.path.join(model_dir, "tree_packed.npz"),
                            blob=blob, meta=np.array([json.dumps(meta_tbl)], dtype=object))
        return

    # ---- compress == "none": legacy raw files per node/layer (fast) ----
    for node in tree_mlp.node_list:
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        save_model(model=node.net, model_path=model_path, storage_conf=S)




def load_tree_models(model_dir:str):
    opt_path = os.path.join(model_dir, 'opt.yaml')
    opt = OmegaConf.load(opt_path)

    # storage meta
    meta_path = os.path.join(model_dir, "storage_meta.json")
    S = {"mode":"int8_tensor","compress":"none","percentile":0.995}
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

    tree_mlp = OctTreeMLP(opt, origin_shape=origin_shape, load_data=False,
                        model_dir=model_dir, defer_build=True)

    tree_mlp.model_dir = model_dir

    compress = S.get("compress","none")
    mode = S.get("mode","int8_tensor")

    if compress == "npz" and os.path.exists(os.path.join(model_dir, "tree_packed.npz")):
        z = np.load(os.path.join(model_dir, "tree_packed.npz"), allow_pickle=True)
        meta_tbl = json.loads(str(z["meta"][0]))
        arrays = _unpack_tensors_from_blob(z["blob"], meta_tbl)

        def _wkey(node_id, li): return f"{node_id}|{li}-W.q8" if mode.startswith("int8") else f"{node_id}|{li}-W"
        def _bkey(node_id, li): return f"{node_id}|{li}-B.q8" if mode.startswith("int8") else f"{node_id}|{li}-B"

        def _infer_dims_from_meta(node_id, layers, in_dim_expected):
            first = meta_tbl[_wkey(node_id, 0)]["shape"]  # (out, in)
            hidden, in_dim = int(first[0]), int(first[1])
            if in_dim_expected is not None and in_dim != int(in_dim_expected):
                raise ValueError(f"in_dim mismatch for {node_id}: archive={in_dim}, expected={in_dim_expected}")
            last_bk = _bkey(node_id, layers-1)
            if last_bk in meta_tbl:
                out_dim = int(meta_tbl[last_bk]["shape"][0])
            else:
                last_wk = _wkey(node_id, layers-1)
                out_dim = int(meta_tbl[last_wk]["shape"][0])
            return hidden, out_dim

        # Fill per-level settings (act/layer/output_act) without constructing nets
        from types import SimpleNamespace  # put this at the top of the file if not already imported

        # Fill per-level settings (act/layer/output_act) without constructing nets
        tree_mlp.get_hyper()

        def _attach_hypers(n):
            layer, act = tree_mlp.level_layer[n.level], tree_mlp.level_act[n.level]
            # store hyper fields only; no network is built here
            n.net = SimpleNamespace(hyper={
                "input": None,          # will be provided by parent during rebuild
                "output": None,         # inferred from archive meta
                "hidden": None,         # inferred from archive meta
                "layer": int(layer),
                "act": act,
                "output_act": bool(n.level < tree_mlp.max_level),
                "w0": int(tree_mlp.opt.Network.w0),
            })
            for ch in n.children:
                _attach_hypers(ch)

        _attach_hypers(tree_mlp.base_node)


        def _rebuild_and_load(node, in_dim_expected):
            node_id = f"{node.level}-{node.di}-{node.hi}-{node.wi}"
            hconf = node.net.hyper
            layers = int(hconf["layer"])

            hidden, out_dim = _infer_dims_from_meta(node_id, layers, in_dim_expected)

            # build exact MLP
            model = MLP(input=int(in_dim_expected), output=int(out_dim), hidden=int(hidden),
                        layer=layers, act=hconf["act"], output_act=bool(hconf["output_act"]), w0=int(hconf["w0"]))

            # load weights
            for li in range(layers):
                layer = model.net[li]
                W_shape = layer[0].weight.shape
                B_shape = layer[0].bias.shape
                if mode in ("fp32","fp16"):
                    W = arrays[_wkey(node_id, li)].astype(np.float32, copy=False).reshape(W_shape)
                    B = arrays[_bkey(node_id, li)].astype(np.float32, copy=False).reshape(B_shape)
                elif mode == "int8_tensor":
                    qW = arrays[_wkey(node_id, li)].astype(np.int8, copy=False).reshape(W_shape)
                    qB = arrays[_bkey(node_id, li)].astype(np.int8, copy=False).reshape(B_shape)
                    sW = arrays[f"{node_id}|{li}-W.scale"].astype(np.float32, copy=False).ravel()[0]
                    sB = arrays[f"{node_id}|{li}-B.scale"].astype(np.float32, copy=False).ravel()[0]
                    W = _dequant(qW, sW); B = _dequant(qB, sB)
                else:  # int8_per_channel
                    qW = arrays[_wkey(node_id, li)].astype(np.int8, copy=False).reshape(W_shape)
                    sW = arrays[f"{node_id}|{li}-W.scale"].astype(np.float32, copy=False)
                    if sW.ndim != 1 or sW.shape[0] != W_shape[0]:
                        raise ValueError("Bad per-channel scale shape")
                    qB = arrays[_bkey(node_id, li)].astype(np.int8, copy=False).reshape(B_shape)
                    sB = arrays[f"{node_id}|{li}-B.scale"].astype(np.float32, copy=False).ravel()[0]
                    W = _dequant(qW, sW); B = _dequant(qB, sB)

                with torch.no_grad():
                    layer[0].weight.data = torch.tensor(W, dtype=torch.float32)
                    layer[0].bias.data   = torch.tensor(B, dtype=torch.float32)

            node.net = model
            for ch in node.children:
                _rebuild_and_load(ch, out_dim)

        _rebuild_and_load(tree_mlp.base_node, int(opt.Network.input))

        # finalize minimal runtime state for decode
        tree_mlp.init_node_list()
        tree_mlp.cal_params_total()
        tree_mlp.move2device(tree_mlp.device)
        tree_mlp.sampler = tree_mlp.init_sampler()

    else:
        # fallback: raw per-node files (compress == "none")
        from types import SimpleNamespace  # put at top of file if not present

        # For compress == "none": attach per-node hyper placeholders (no MLP constructed yet)
        tree_mlp.get_hyper()

        def _attach_hypers_raw(n):
            layer, act = tree_mlp.level_layer[n.level], tree_mlp.level_act[n.level]
            n.net = SimpleNamespace(hyper={
                "input": None,          # provided by parent during recursion
                "output": None,         # inferred by load_model_from_files
                "hidden": None,         # inferred by load_model_from_files
                "layer": int(layer),
                "act": act,
                "output_act": bool(n.level < tree_mlp.max_level),
                "w0": int(tree_mlp.opt.Network.w0),
            })
            for ch in n.children:
                _attach_hypers_raw(ch)

        _attach_hypers_raw(tree_mlp.base_node)

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
        tree_mlp.init_node_list()
        tree_mlp.cal_params_total()
        tree_mlp.move2device(tree_mlp.device)
        tree_mlp.sampler = tree_mlp.init_sampler()

    if saved_stats:
        if "dtype" in saved_stats:
            saved_stats["dtype"] = np.dtype(saved_stats["dtype"])
        tree_mlp.side_info.update(saved_stats)

    return tree_mlp

