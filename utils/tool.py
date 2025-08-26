import cv2
import tifffile
import os
import numpy as np

# def get_type_max(data):
#     dtype = data.dtype.name
#     if dtype == 'uint8':
#         max = 255
#     elif dtype == 'uint12':
#         max = 4098
#     elif dtype == 'uint16':
#         max = 65535
#     elif dtype == 'float32':
#         max = 65535
#     elif dtype == 'float64':
#         max = 65535
#     elif dtype == 'int16':
#         max = 65535   
#     else:
#         raise NotImplementedError
#     return max

def get_type_max(data: np.ndarray) -> float:
    dt = data.dtype

    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        vmin = int(np.min(data))
        vmax = int(np.max(data))

        # Unsigned ints
        if info.min == 0:
            # Heuristic: 12-bit content stored in uint16
            if info.bits == 16 and vmax <= 4095:
                return 4095.0
            return float(info.max)

        # Signed ints
        if vmin >= 0:
            # data never goes negative → treat like unsigned
            return float(info.max)
        # otherwise use full value range as the peak
        return float(info.max - info.min)

    if np.issubdtype(dt, np.floating):
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        # Common conventions
        if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
            return 1.0
        if vmin >= -1e-6 and vmax <= 100.0 + 1e-6:
            return 100.0
        # Fallback: use the actual max magnitude seen
        return max(abs(vmax), 1.0)

    raise NotImplementedError(f"Unsupported dtype: {dt}")

# 3d->dhwc or thwc 2d->hwc
def read_img(path):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        img = tifffile.imread(path)
        if len(img.shape) == 3:
            img = img[...,None]
        assert len(img.shape)==4
    elif postfix in ['.png','.jpg']:
        img = cv2.imread(path,-1)
        if len(img.shape) == 2:
            img = img[...,None]
        assert len(img.shape)==3
    else:
        raise NotImplemented
    return img  

def save_img(path, img):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        tifffile.imsave(path, img)
    elif postfix in ['.png','.jpg']:
        cv2.imwrite(path, img)  
    else:
        raise NotImplemented  

def get_folder_size(folder_path:str):
    total_size = 0
    if os.path.isdir(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(folder_path)
    return total_size