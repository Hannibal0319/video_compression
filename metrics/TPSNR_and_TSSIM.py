import cv2
import numpy as np
import concurrent.futures
import os
import subprocess
import tempfile
from uuid import uuid4
from skimage.metrics import structural_similarity as ssim
from load_video_frames import load_video_frames

_FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
_NUL = "NUL" if os.name == "nt" else "/dev/null"


def _to_gray_list(frames, num, scale_factor=1.0, luma_only=True):
    out = []
    for f in frames[:num]:
        if scale_factor != 1.0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY if luma_only else cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(gray)
    return out


def _calc_flow(prev, curr):
    try:
        return cv2.calcOpticalFlowFarneback(prev, curr, None, **_FLOW_PARAMS)
    except cv2.error:
        return None


def _warp(prev, flow, grid_x, grid_y, w, h):
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    return cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _worker_pair(idx, orig_gray, comp_gray, grid_x, grid_y, w, h, single_flow):
    o_prev = orig_gray[idx - 1]; o_curr = orig_gray[idx]
    c_prev = comp_gray[idx - 1]; c_curr = comp_gray[idx]

    flow_o = _calc_flow(o_prev, o_curr)
    if flow_o is None:
        return None

    flow_c = None if single_flow else _calc_flow(c_prev, c_curr)
    if flow_c is None and not single_flow:
        return None

    warped_o = _warp(o_prev, flow_o, grid_x, grid_y, w, h)
    if flow_c is None:
        warped_c = _warp(c_prev, flow_o, grid_x, grid_y, w, h)
    else:
        warped_c = _warp(c_prev, flow_c, grid_x, grid_y, w, h)

    # PSNR
    diff = warped_o.astype(np.float32) - warped_c.astype(np.float32)
    mse = np.mean(diff ** 2)
    if not np.isfinite(mse) or mse < 0:
        return None
    psnr_val = float('inf') if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)

    # SSIM: ensure proper dtype/range
    wo_uint8 = np.clip(warped_o, 0, 255).astype(np.uint8)
    wc_uint8 = np.clip(warped_c, 0, 255).astype(np.uint8)
    try:
        ssim_val = ssim(wo_uint8, wc_uint8, data_range=255)
    except Exception:
        return None
    if not np.isfinite(ssim_val):
        return None

    return float(ssim_val), float(psnr_val)


def _run_pairs(orig_frames, comp_frames, single_flow=False, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):
    ssim_values = []
    psnr_values = []
    num_frames = min(len(orig_frames), len(comp_frames))
    if num_frames < 2:
        return ssim_values, psnr_values

    orig_gray = _to_gray_list(orig_frames, num_frames, scale_factor, luma_only)
    comp_gray = _to_gray_list(comp_frames, num_frames, scale_factor, luma_only)

    h, w = orig_gray[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    indices = list(range(1, num_frames, frame_step))
    cpu_count = os.cpu_count() or 4
    max_workers = min(len(indices), max(1, cpu_count))
    Executor = concurrent.futures.ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker_pair, idx, orig_gray, comp_gray, grid_x, grid_y, w, h, single_flow): idx for idx in indices}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            s_val, p_val = res
            ssim_values.append(s_val)
            psnr_values.append(p_val)

    return ssim_values, psnr_values


def compute_temporal_psnr(orig_frames, comp_frames, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=False, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)
    if len(psnr_vals) == 0:
        return float('nan')
    return float(np.mean(psnr_vals))



def compute_temporal_SSIM(orig_frames, comp_frames, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=False, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)
    if len(ssim_vals) == 0:
        return float('nan')
    return float(np.mean(ssim_vals))


def compute_tSSIM_and_tPSNR(orig_frames, comp_frames, single_flow=False, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=single_flow, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)
    if len(ssim_vals) == 0 or len(psnr_vals) == 0:
        return float('nan'), float('nan')
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))

def tSSIM_by_paths(orig_video_path, comp_video_path, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):

    orig_frames = load_video_frames(orig_video_path)
    comp_frames = load_video_frames(comp_video_path)
    return compute_temporal_SSIM(orig_frames, comp_frames, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)

def tPSNR_by_paths(orig_video_path, comp_video_path, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):

    orig_frames = load_video_frames(orig_video_path)
    comp_frames = load_video_frames(comp_video_path)
    return compute_temporal_psnr(orig_frames, comp_frames, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)

import time



def compute_tSSIM_and_tPSNR_by_paths(orig_video_path, comp_video_path, frame_step=1, scale_factor=1.0, luma_only=True, use_process_pool=False):
    start_time = time.time()
    orig_frames = load_video_frames(orig_video_path)
    print(f"Loaded original video frames in {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    comp_frames = load_video_frames(comp_video_path)
    print(f"Loaded compressed video frames in {time.time() - start_time:.2f} seconds")
    return compute_tSSIM_and_tPSNR(orig_frames, comp_frames, frame_step=frame_step, scale_factor=scale_factor, luma_only=luma_only, use_process_pool=use_process_pool)
