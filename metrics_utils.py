import cv2
import numpy as np

# Compatibility shim for packages (e.g. scikit-video) that still use deprecated
# numpy aliases like np.float, np.int, etc. Restores safe aliases when missing.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str

from skimage.metrics import structural_similarity as ssim
#from pytorch_msssim import ms_ssim, MS_SSIM
import torch
import time
import concurrent.futures
import os

def compute_temporal_psnr(orig_frames, comp_frames):
    psnr_values = []
    num_frames = min(len(orig_frames), len(comp_frames))

    if num_frames < 2:
        # Not enough frames to compute temporal metrics
        return float('nan')

    # Pre-convert to grayscale once (float32)
    orig_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in orig_frames[:num_frames]]
    comp_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in comp_frames[:num_frames]]

    # assume all frames same size; use first for grid
    h, w = orig_gray[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    def process_pair(idx):
        # compute optical flow and return psnr or None on failure
        orig_prev = orig_gray[idx-1]
        orig_curr = orig_gray[idx]
        comp_prev = comp_gray[idx-1]
        comp_curr = comp_gray[idx]

        try:
            flow_orig = cv2.calcOpticalFlowFarneback(orig_prev, orig_curr, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow_comp = cv2.calcOpticalFlowFarneback(comp_prev, comp_curr, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        except cv2.error:
            return None

        # warp using flow_orig and flow_comp
        map_x = (grid_x + flow_orig[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_orig[..., 1]).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        warped_orig = cv2.remap(orig_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        map_x = (grid_x + flow_comp[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_comp[..., 1]).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        warped_comp = cv2.remap(comp_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        diff = warped_orig.astype(np.float32) - warped_comp.astype(np.float32)
        mse = np.mean(diff ** 2)
        if not np.isfinite(mse) or mse < 0:
            return None
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10((255.0 ** 2) / mse)
        return float(psnr)

    indices = list(range(1, num_frames))
    max_workers = min(len(indices), max(1, (os.cpu_count()-4 or 4)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_pair, idx): idx for idx in indices}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            psnr_values.append(res)

    if len(psnr_values) == 0:
        return float('nan')
    avg_psnr = float(np.mean(psnr_values))
    return avg_psnr

def compute_temporal_SSIM(orig_frames, comp_frames):
    ssim_values = []
    num_frames = min(len(orig_frames), len(comp_frames))

    if num_frames < 2:
        return float('nan')

    # Pre-convert to grayscale once (float32)
    orig_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in orig_frames[:num_frames]]
    comp_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in comp_frames[:num_frames]]

    # assume all frames same size; use first for grid
    h, w = orig_gray[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    def process_pair(idx):
        orig_prev = orig_gray[idx - 1]
        orig_curr = orig_gray[idx]
        comp_prev = comp_gray[idx - 1]
        comp_curr = comp_gray[idx]

        try:
            flow_orig = cv2.calcOpticalFlowFarneback(orig_prev, orig_curr, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow_comp = cv2.calcOpticalFlowFarneback(comp_prev, comp_curr, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        except cv2.error:
            return None

        map_x = (grid_x + flow_orig[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_orig[..., 1]).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        warped_orig = cv2.remap(orig_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        map_x = (grid_x + flow_comp[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_comp[..., 1]).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        warped_comp = cv2.remap(comp_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Ensure values are in 0-255 range and uint8 for ssim
        warped_orig_clipped = np.clip(warped_orig, 0, 255).astype(np.uint8)
        warped_comp_clipped = np.clip(warped_comp, 0, 255).astype(np.uint8)

        try:
            ssim_index = ssim(warped_orig_clipped, warped_comp_clipped, data_range=255)
        except Exception:
            return None
        if not np.isfinite(ssim_index):
            return None
        return float(ssim_index)

    indices = list(range(1, num_frames))
    max_workers = min(len(indices), max(1, (os.cpu_count() or 4)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_pair, idx): idx for idx in indices}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            ssim_values.append(res)

    if len(ssim_values) == 0:
        return float('nan')
    avg_ssim = float(np.mean(ssim_values))
    return avg_ssim

def compute_tSSIM_and_tPSNR(orig_frames, comp_frames, single_flow=False):
    """
    Compute temporal SSIM and temporal PSNR in a single pass reusing optical-flow + warping
    to avoid duplicated work in separate functions. Returns (temporal_ssim, temporal_psnr).

    single_flow: if True compute optical flow only on the original pair and reuse that flow
                 to warp the compressed previous frame as well (faster, less accurate).
    """
    ssim_values = []
    psnr_values = []
    num_frames = min(len(orig_frames), len(comp_frames))

    if num_frames < 2:
        return float('nan'), float('nan')

    # Pre-convert to grayscale once (float32)
    orig_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in orig_frames[:num_frames]]
    comp_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in comp_frames[:num_frames]]

    h, w = orig_gray[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    def process_pair(idx):
        orig_prev = orig_gray[idx - 1]
        orig_curr = orig_gray[idx]
        comp_prev = comp_gray[idx - 1]
        comp_curr = comp_gray[idx]

        try:
            # always compute flow on the original pair
            flow_orig = cv2.calcOpticalFlowFarneback(orig_prev, orig_curr, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            if single_flow:
                flow_comp = None
            else:
                flow_comp = cv2.calcOpticalFlowFarneback(comp_prev, comp_curr, None,
                                                         pyr_scale=0.5, levels=3, winsize=15,
                                                         iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        except cv2.error:
            return None

        # warp previous original using flow_orig
        map_x = (grid_x + flow_orig[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_orig[..., 1]).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        warped_orig = cv2.remap(orig_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # warp previous compressed using either its own flow or the original flow (faster)
        if flow_comp is None:
            map_x_c = map_x
            map_y_c = map_y
        else:
            map_x_c = (grid_x + flow_comp[..., 0]).astype(np.float32)
            map_y_c = (grid_y + flow_comp[..., 1]).astype(np.float32)
            map_x_c = np.clip(map_x_c, 0, w - 1)
            map_y_c = np.clip(map_y_c, 0, h - 1)
        warped_comp = cv2.remap(comp_prev, map_x_c, map_y_c, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # PSNR (float32)
        diff = warped_orig.astype(np.float32) - warped_comp.astype(np.float32)
        mse = np.mean(diff ** 2)
        if not np.isfinite(mse) or mse < 0:
            return None
        psnr = float('inf') if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)

        # SSIM (skimage expects uint8 or float with proper data_range)
        warped_orig_clipped = np.clip(warped_orig, 0, 255).astype(np.uint8)
        warped_comp_clipped = np.clip(warped_comp, 0, 255).astype(np.uint8)
        try:
            ssim_index = ssim(warped_orig_clipped, warped_comp_clipped, data_range=255)
        except Exception:
            return None
        if not np.isfinite(ssim_index):
            return None

        return float(ssim_index), float(psnr)

    indices = list(range(1, num_frames))
    cpu_count = os.cpu_count() or 4
    max_workers = min(len(indices), max(1, cpu_count))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_pair, idx): idx for idx in indices}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            s_val, p_val = res
            ssim_values.append(s_val)
            psnr_values.append(p_val)

    if len(ssim_values) == 0 or len(psnr_values) == 0:
        return float('nan'), float('nan')

    avg_ssim = float(np.mean(ssim_values))
    avg_psnr = float(np.mean(psnr_values))
    return avg_ssim, avg_psnr

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

    temporal_ssim, temporal_psnr = compute_tSSIM_and_tPSNR(orig_frames, comp_frames)
    return temporal_ssim, temporal_psnr

def MS_SSIM(orig_frames, comp_frames):
    ms_ssim_module = MS_SSIM(data_range=len(orig_frames), size_average=True, channel=3)
    ms_ssim_value = ms_ssim_module(
        torch.from_numpy(orig_frames).permute(0, 3, 1, 2).float(),
        torch.from_numpy(comp_frames).permute(0, 3, 1, 2).float(),
    )
    return ms_ssim_value.item()

def tPSNR_by_paths(orig_video_path, comp_video_path, length=100, single_flow=True):
    orig_cap = cv2.VideoCapture(orig_video_path)
    comp_cap = cv2.VideoCapture(comp_video_path)

    orig_frames = []
    comp_frames = []
    i = 0
    while True:
        ret_orig, frame_orig = orig_cap.read()
        ret_comp, frame_comp = comp_cap.read()
        if not ret_orig or not ret_comp or i >= length:
            break
        orig_frames.append(frame_orig)
        comp_frames.append(frame_comp)
        i += 1

    orig_cap.release()
    comp_cap.release()

    _, temporal_psnr = compute_tSSIM_and_tPSNR(orig_frames, comp_frames, single_flow=single_flow)
    return temporal_psnr

def tSSIM_by_paths(orig_video_path, comp_video_path, length=100, single_flow=True):
    orig_cap = cv2.VideoCapture(orig_video_path)
    comp_cap = cv2.VideoCapture(comp_video_path)

    orig_frames = []
    comp_frames = []
    i = 0
    while True:
        ret_orig, frame_orig = orig_cap.read()
        ret_comp, frame_comp = comp_cap.read()
        if not ret_orig or not ret_comp or i >= length:
            break
        orig_frames.append(frame_orig)
        comp_frames.append(frame_comp)
        i += 1

    orig_cap.release()
    comp_cap.release()

    temporal_ssim, _ = compute_tSSIM_and_tPSNR(orig_frames, comp_frames, single_flow=single_flow)
    return temporal_ssim

def ST_RRED_by_paths(orig_video_path, comp_video_path):
    print("Reading videos for ST-RRED computation...")
    videodata_ref = []
    videodata_dist = []
    cap_ref = cv2.VideoCapture(orig_video_path)
    cap_dist = cv2.VideoCapture(comp_video_path)
    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_dist, frame_dist = cap_dist.read()
        if not ret_ref or not ret_dist:
            break
        videodata_ref.append(frame_ref)
        videodata_dist.append(frame_dist)
    cap_ref.release()
    cap_dist.release()
    videodata_ref = np.array(videodata_ref)
    videodata_dist = np.array(videodata_dist)

    print("Computing ST-RRED...")
    videodata_ref = np.squeeze(videodata_ref)
    videodata_dist = np.squeeze(videodata_dist)
    st_rred = compute_strred(videodata_ref, videodata_dist)
    return st_rred

def MS_SSIM_by_paths(orig_video_path, comp_video_path):
    try:
        videodata_ref = []
        videodata_dist = []
        cap_ref = cv2.VideoCapture(orig_video_path)
        cap_dist = cv2.VideoCapture(comp_video_path)
        while True:
            ret_ref, frame_ref = cap_ref.read()
            ret_dist, frame_dist = cap_dist.read()
            if not ret_ref or not ret_dist:
                break
            videodata_ref.append(frame_ref)
            videodata_dist.append(frame_dist)
        cap_ref.release()
        cap_dist.release()
        videodata_ref = np.array(videodata_ref)
        videodata_dist = np.array(videodata_dist)
    except Exception as e:
        print(f"Error reading videos for MS-SSIM: {e}")
        return float('nan')
    
    ms_ssim_value = compute_ms_ssim(videodata_ref, videodata_dist)


    return float(ms_ssim_value)

def compute_ms_ssim(orig_frames, comp_frames):
    pass  # Placeholder

import pywt

def compute_strred(orig_frames, comp_frames):
    def spatial_entropy(frame):
        coeffs = pywt.dwt2(frame, 'db1')
        LH, HL, HH = coeffs[1]
        entropy = np.mean(np.abs(LH)) + np.mean(np.abs(HL)) + np.mean(np.abs(HH))
        return entropy
    def temporal_entropy(frame1, frame2):
        diff = np.abs(frame2.astype(float) - frame1.astype(float))
        return np.mean(diff)
    spatial_scores = []
    temporal_scores = []
    print("Calculating spatial and temporal entropies for ST-RRED...")
    #compute spatial entropy for each frame with multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        spatial_futures = {executor.submit(spatial_entropy, comp_frames[i]): i for i in range(len(comp_frames))}
        for fut in concurrent.futures.as_completed(spatial_futures):
            res = fut.result()
            spatial_scores.append(res)
    #compute temporal entropy for each consecutive frame pair with multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        temporal_futures = {executor.submit(temporal_entropy, comp_frames[i-1], comp_frames[i]): i for i in range(1, len(comp_frames))}
        for fut in concurrent.futures.as_completed(temporal_futures):
            res = fut.result()
            temporal_scores.append(res)
    ST_RRED = np.mean(spatial_scores) + np.mean(temporal_scores)
    return ST_RRED

def csf_weight(frame):
    return np.std(frame)  # perceptual contrast weight

def movie_s(frame_ref, frame_dist):
    # ensure single-channel float arrays for wavelet transform
    if frame_ref.ndim == 3:
        frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        frame_ref_gray = frame_ref.astype(np.float32)
    if frame_dist.ndim == 3:
        frame_dist_gray = cv2.cvtColor(frame_dist, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        frame_dist_gray = frame_dist.astype(np.float32)

    coeffs_ref = pywt.dwt2(frame_ref_gray, 'db1')
    coeffs_dist = pywt.dwt2(frame_dist_gray, 'db1')
    LH_r, HL_r, HH_r = coeffs_ref[1]
    LH_d, HL_d, HH_d = coeffs_dist[1]
    return csf_weight(frame_ref_gray) * (
        abs(np.mean(LH_r) - np.mean(LH_d)) +
        abs(np.mean(HL_r) - np.mean(HL_d)) +
        abs(np.mean(HH_r) - np.mean(HH_d))
    )

def movie_t(frame1_ref, frame2_ref, frame1_dist, frame2_dist):
    # prepare single-channel float32 frames and ensure matching sizes
    def prep(frame):
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return gray.astype(np.float32)

    f1r = prep(frame1_ref)
    f2r = prep(frame2_ref)
    f1d = prep(frame1_dist)
    f2d = prep(frame2_dist)

    # resize distorted frames to match reference sizes if needed
    h, w = f1r.shape
    if f2r.shape != (h, w):
        f2r = cv2.resize(f2r, (w, h), interpolation=cv2.INTER_LINEAR)
    if f1d.shape != (h, w):
        f1d = cv2.resize(f1d, (w, h), interpolation=cv2.INTER_LINEAR)
    if f2d.shape != (h, w):
        f2d = cv2.resize(f2d, (w, h), interpolation=cv2.INTER_LINEAR)

    flow_ref = cv2.calcOpticalFlowFarneback(f1r, f2r, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow_dist = cv2.calcOpticalFlowFarneback(f1d, f2d, None,
                                             pyr_scale=0.5, levels=3, winsize=15,
                                             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return float(np.mean(np.abs(flow_ref - flow_dist)))

def compute_movie_index(orig_frames, comp_frames):
    spatial_scores = []
    temporal_scores = []
    print("Calculating movie index spatial scores...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        spatial_futures = {executor.submit(movie_s, orig_frames[i], comp_frames[i]): i for i in range(len(comp_frames))}
        for fut in concurrent.futures.as_completed(spatial_futures):
            res = fut.result()
            spatial_scores.append(res)
    end_time = time.time()
    print(f"Spatial scores computed in {end_time - start_time:.2f} seconds")
    start_time = time.time()
    print("Calculating movie index temporal scores...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        temporal_futures = {executor.submit(movie_t, orig_frames[i-1], orig_frames[i],
                                           comp_frames[i-1], comp_frames[i]): i for i in range(1, len(comp_frames))}
        for fut in concurrent.futures.as_completed(temporal_futures):
            res = fut.result()
            temporal_scores.append(res)
    end_time = time.time()
    print(f"Temporal scores computed in {end_time - start_time:.2f} seconds")
    movie_index = np.mean(spatial_scores) + np.mean(temporal_scores)
    return movie_index

def compute_movie_index_by_paths(orig_video_path, comp_video_path):
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

    movie_index = compute_movie_index(orig_frames, comp_frames)
    return movie_index
    

if __name__ == "__main__":
    # Example usage
    orig_video_path = "videos/UVG/Jockey_1920x1080_120fps_420_8bit_YUV.y4m"
    comp_video_path = "compressed_videos\\UVG\\h264\\1\\Jockey_1920x1080_120fps_420_8bit_YUV_h264.mp4"
    start_time = time.time()
    movie_val = compute_movie_index_by_paths(orig_video_path, comp_video_path)
    end_time = time.time()
    print(f"Movie Index: {movie_val}, computed in {end_time - start_time:.2f} seconds")