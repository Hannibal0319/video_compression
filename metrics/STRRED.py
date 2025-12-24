from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import pywt
import concurrent.futures


def compute_strred(orig_frames, comp_frames):
    def calculate_st_entropy(frames):
        """Helper to calculate ST entropy for a set of frames."""
        def spatial_entropy(frame):
            # Convert to grayscale if it's a color image
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coeffs = pywt.dwt2(frame, 'db1')
            LH, HL, HH = coeffs[1]
            entropy = np.mean(np.abs(LH)) + np.mean(np.abs(HL)) + np.mean(np.abs(HH))
            return entropy

        def temporal_entropy(frame1, frame2):
            # Convert to grayscale if it's a color image
            if frame1.ndim == 3 and frame1.shape[2] == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if frame2.ndim == 3 and frame2.shape[2] == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            diff = np.abs(frame2.astype(float) - frame1.astype(float))
            return np.mean(diff)

        spatial_scores = []
        temporal_scores = []
        num_frames = len(frames)
        if num_frames == 0:
            return 0

        max_workers = os.cpu_count() - 4 or 4
        
        # Compute spatial entropy for each frame
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            spatial_futures = {executor.submit(spatial_entropy, frames[i]): i for i in range(num_frames)}
            for fut in concurrent.futures.as_completed(spatial_futures):
                spatial_scores.append(fut.result())

        # Compute temporal entropy for each consecutive frame pair
        if num_frames > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                temporal_futures = {executor.submit(temporal_entropy, frames[i-1], frames[i]): i for i in range(1, num_frames)}
                for fut in concurrent.futures.as_completed(temporal_futures):
                    temporal_scores.append(fut.result())

        mean_spatial = np.mean(spatial_scores) if spatial_scores else 0
        mean_temporal = np.mean(temporal_scores) if temporal_scores else 0
        
        return mean_spatial + mean_temporal

    print("Calculating ST-RRED for original video...")
    st_rred_orig = calculate_st_entropy(orig_frames)
    
    print("Calculating ST-RRED for compressed video...")
    st_rred_comp = calculate_st_entropy(comp_frames)

    # The final metric is the difference between the two entropy scores
    st_rred_diff = abs(st_rred_orig - st_rred_comp)
    
    return st_rred_diff

import torch
import torch.nn.functional as F
 
def compute_strred_gpu(orig_frames, comp_frames, device=None):
    """
    Compute ST-RRED using PyTorch. If a CUDA GPU is available, computations run there.

    Args:
        orig_frames (List[np.ndarray]): Original video frames (BGR or gray).
        comp_frames (List[np.ndarray]): Compressed video frames.
        device (str, optional): Torch device override. Default: 'cuda' if available else 'cpu'.

    Returns:
        float: Absolute difference between original and compressed ST-RRED scores.
    """

    def frames_to_tensor(frames, device):
        if len(frames) == 0:
            return None
        arr = np.stack(frames)  # (N, H, W[, 3])
        tensor = torch.from_numpy(arr).float().to(device)
        if tensor.ndim == 4 and tensor.shape[-1] == 3:
            # Convert BGR to grayscale with luma weights
            b, g, r = tensor.unbind(-1)
            tensor = 0.114 * b + 0.587 * g + 0.299 * r
        tensor = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor.unsqueeze(1)  # (N, 1, H, W)
        return tensor

    def batch_dwt2(x):
        # Single-level 2D Haar DWT implemented with conv2d; reflection pad matches pywt symmetric padding.
        # Keep filter dtype aligned with input to avoid torch type mismatch on CUDA/CPU.
        h0 = torch.tensor([1 / np.sqrt(2), 1 / np.sqrt(2)], device=x.device, dtype=x.dtype)
        h1 = torch.tensor([-1 / np.sqrt(2), 1 / np.sqrt(2)], device=x.device, dtype=x.dtype)
        filters = torch.stack(
            [
                torch.outer(h0, h0),  # LL (unused)
                torch.outer(h1, h0),  # LH
                torch.outer(h0, h1),  # HL
                torch.outer(h1, h1),  # HH
            ],
            dim=0,
        ).unsqueeze(1)  # (4, 1, 2, 2)

        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        coeffs = F.conv2d(x_padded, filters, stride=2)
        # Preserve a channel dimension so downstream mean dims stay consistent (N, C, H, W).
        return coeffs[:, 1:2], coeffs[:, 2:3], coeffs[:, 3:4]

    def calculate_st_entropy(frames, device):
        tensor = frames_to_tensor(frames, device)
        if tensor is None:
            return 0.0

        LH, HL, HH = batch_dwt2(tensor)
        spatial_entropy = (LH.abs().mean(dim=(1, 2, 3)) + HL.abs().mean(dim=(1, 2, 3)) + HH.abs().mean(dim=(1, 2, 3))).mean().item()

        temporal_entropy = 0.0
        if tensor.size(0) > 1:
            diff = (tensor[1:] - tensor[:-1]).abs().mean(dim=(1, 2, 3))
            temporal_entropy = diff.mean().item()

        return spatial_entropy + temporal_entropy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Calculating ST-RRED for original video on {device}...")
    st_rred_orig = calculate_st_entropy(orig_frames, device)

    print(f"Calculating ST-RRED for compressed video on {device}...")
    st_rred_comp = calculate_st_entropy(comp_frames, device)

    st_rred_diff = abs(st_rred_orig - st_rred_comp)
    return st_rred_diff

import time

def load_video_frames(video_path, max_frames=None):
    """Load video frames from a file into a list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return frames

if __name__ == "__main__":
    # Example usage with dummy data
    orig_frames = load_video_frames("videos/BVI-HD/BallUnderWater_1920x1080_60fps.mp4")
    comp_frames = load_video_frames("compressed_videos/BVI-HD/h264/1/BallUnderWater_1920x1080_60fps_h264.mp4")
    print(f"Loaded {len(orig_frames)} original frames and {len(comp_frames)} compressed frames.")
    start_time = time.time()
    strred_cpu = compute_strred(orig_frames, comp_frames)
    end_time = time.time()
    print(f"ST-RRED (CPU): {strred_cpu}, Elapsed time: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    strred_gpu = compute_strred_gpu(orig_frames, comp_frames)
    end_time = time.time()

    print(f"ST-RRED (GPU): {strred_gpu}, Elapsed time: {end_time - start_time:.2f} seconds")