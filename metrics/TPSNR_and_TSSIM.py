import cv2
import numpy as np
import concurrent.futures
import os
from skimage.metrics import structural_similarity as ssim


_FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _to_gray_list(frames, num):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames[:num]]


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


def _run_pairs(orig_frames, comp_frames, single_flow=False):
    ssim_values = []
    psnr_values = []
    num_frames = min(len(orig_frames), len(comp_frames))
    if num_frames < 2:
        return ssim_values, psnr_values

    orig_gray = _to_gray_list(orig_frames, num_frames)
    comp_gray = _to_gray_list(comp_frames, num_frames)

    h, w = orig_gray[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    indices = list(range(1, num_frames))
    cpu_count = os.cpu_count() or 4
    max_workers = min(len(indices), max(1, cpu_count))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker_pair, idx, orig_gray, comp_gray, grid_x, grid_y, w, h, single_flow): idx for idx in indices}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            s_val, p_val = res
            ssim_values.append(s_val)
            psnr_values.append(p_val)

    return ssim_values, psnr_values


def compute_temporal_psnr(orig_frames, comp_frames):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=False)
    if len(psnr_vals) == 0:
        return float('nan')
    return float(np.mean(psnr_vals))


def compute_temporal_SSIM(orig_frames, comp_frames):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=False)
    if len(ssim_vals) == 0:
        return float('nan')
    return float(np.mean(ssim_vals))


def compute_tSSIM_and_tPSNR(orig_frames, comp_frames, single_flow=False):
    ssim_vals, psnr_vals = _run_pairs(orig_frames, comp_frames, single_flow=single_flow)
    if len(ssim_vals) == 0 or len(psnr_vals) == 0:
        return float('nan'), float('nan')
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def compute_tSSIM_and_tPSNR_by_paths(orig_video_path, comp_video_path):
    orig_cap = cv2.VideoCapture(orig_video_path)
    comp_cap = cv2.VideoCapture(comp_video_path)

    orig_frames = []
    comp_frames = []
    while True:
        ret_orig, frame_orig = orig_cap.read()
        ret_comp, frame_comp = comp_cap.read()
        if not ret_orig or not ret_comp:
            break

        orig_frames.append(frame_orig)
        comp_frames.append(frame_comp)

    orig_cap.release()
    comp_cap.release()

    return compute_tSSIM_and_tPSNR(orig_frames, comp_frames)