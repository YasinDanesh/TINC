import os
import cv2
import numpy as np
from utils.tool import get_type_max, read_img, save_img
from omegaconf import OmegaConf
import torch
import json
import sys
from tqdm import tqdm
from einops import rearrange, repeat
from utils.ssim import ssim as ssim_calc
from utils.ssim import ms_ssim as ms_ssim_calc
import copy

def cal_iou_acc_pre(data_gt:np.ndarray,data_hat:np.ndarray,thres:float=1):
    hat = np.copy(data_hat)
    gt = np.copy(data_gt)
    hat[data_hat>=thres]=1
    hat[data_hat<thres]=0
    gt[data_gt>=thres]=1
    gt[data_gt<thres]=0
    tp = (gt*hat).sum()
    tn = ((gt+hat)==0).sum()
    fp = ((gt==0)*(hat==1)).sum()
    fn = ((gt==1)*(hat==0)).sum()
    iou = 1.0*tp/(tp+fp+fn)
    acc = 1.0*(tp+tn)/(tp+fp+tn+fn)
    pre = 1.0*tp/(tp+fp)
    return iou, acc, pre

# def cal_psnr(data_gt:np.ndarray, data_hat:np.ndarray, data_range):
#     data_gt = np.copy(data_gt)
#     data_hat = np.copy(data_hat)
#     mse = np.mean(np.power(data_gt/data_range-data_hat/data_range,2))
#     psnr = -10*np.log10(mse)
#     return psnr

#addd
def cal_psnr(data_gt: np.ndarray, data_hat: np.ndarray, data_range):
    # Standard PSNR: 10*log10(peak^2 / MSE).
    # Do NOT rescale the arrays inside this function—just use the peak you pass in.
    gt  = data_gt.astype(np.float64, copy=False)
    hat = data_hat.astype(np.float64, copy=False)
    mse = np.mean((gt - hat) ** 2, dtype=np.float64)
    if not np.isfinite(mse) or mse <= 0.0:
        return float("inf")
    peak = float(data_range)
    return 10.0 * np.log10((peak * peak) / mse)
#adddd

def eval_performance(orig_data, decompressed_data):
    # --- defaults ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8

    # --- accuracy & PSNR (numpy) ---
    max_range = get_type_max(orig_data)
    orig_np   = orig_data.astype(np.float32, copy=False)
    decomp_np = decompressed_data.astype(np.float32, copy=False)

    acc200 = cal_iou_acc_pre(orig_np, decomp_np, thres=200)[1]
    acc500 = cal_iou_acc_pre(orig_np, decomp_np, thres=500)[1]
    psnr_value = cal_psnr(orig_np, decomp_np, max_range)

    # --- SSIM (batched) ---
    with torch.no_grad():
        if orig_np.ndim == 3:
            X = torch.from_numpy(rearrange(orig_np,   'h w (n c) -> n c h w', n=1)).to(device=device, dtype=torch.float32)
            Y = torch.from_numpy(rearrange(decomp_np, 'h w (n c) -> n c h w', n=1)).to(device=device, dtype=torch.float32)
            ssim_value = float(ssim_calc(X, Y, data_range=max_range, size_average=True))
        elif orig_np.ndim == 4:
            Z = orig_np.shape[0]
            ssim_sum = 0.0
            with tqdm(total=Z, desc='Evaluating', position=0, leave=False, dynamic_ncols=True, file=sys.stdout) as pbar:
                for start in range(0, Z, batch_size):
                    end = min(start + batch_size, Z)
                    Xb = torch.from_numpy(rearrange(orig_np[start:end],   '(n) h w c -> n c h w')).to(device=device, dtype=torch.float32)
                    Yb = torch.from_numpy(rearrange(decomp_np[start:end], '(n) h w c -> n c h w')).to(device=device, dtype=torch.float32)
                    chunk = ssim_calc(Xb, Yb, data_range=max_range, size_average=True)  # scalar over this mini-batch
                    ssim_sum += float(chunk) * (end - start)
                    pbar.update(end - start)
            ssim_value = ssim_sum / Z
        else:
            raise ValueError(f'Unexpected shapes: {orig_np.shape} vs {decomp_np.shape}')

    return psnr_value, float(ssim_value), acc200, acc500