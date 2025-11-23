import os
import cv2
import numpy as np
import concurrent.futures
import pywt

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        spatial_futures = {executor.submit(movie_s, orig_frames[i], comp_frames[i]): i for i in range(len(comp_frames))}
        for fut in concurrent.futures.as_completed(spatial_futures):
            res = fut.result()
            spatial_scores.append(res)

    print("Calculating movie index temporal scores...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        temporal_futures = {executor.submit(movie_t, orig_frames[i-1], orig_frames[i],
                                           comp_frames[i-1], comp_frames[i]): i for i in range(1, len(comp_frames))}
        for fut in concurrent.futures.as_completed(temporal_futures):
            res = fut.result()
            temporal_scores.append(res)

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
    
