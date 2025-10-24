import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim, MS_SSIM
import torch


def compute_temporal_psnr(orig_frames, comp_frames):
    psnr_values = []
    num_frames = min(len(orig_frames), len(comp_frames))

    for i in range(1, num_frames):
        orig_prev = cv2.cvtColor(orig_frames[i-1], cv2.COLOR_BGR2GRAY)
        orig_curr = cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2GRAY)
        comp_prev = cv2.cvtColor(comp_frames[i-1], cv2.COLOR_BGR2GRAY)
        comp_curr = cv2.cvtColor(comp_frames[i], cv2.COLOR_BGR2GRAY)

        # Motion estimation (optical flow)
        flow_orig = cv2.calcOpticalFlowFarneback(orig_prev, orig_curr, None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow_comp = cv2.calcOpticalFlowFarneback(comp_prev, comp_curr, None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        h, w = orig_curr.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow_orig[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_orig[..., 1]).astype(np.float32)
        warped_orig = cv2.remap(orig_prev, map_x, map_y, cv2.INTER_LINEAR)

        map_x = (grid_x + flow_comp[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_comp[..., 1]).astype(np.float32)
        warped_comp = cv2.remap(comp_prev, map_x, map_y, cv2.INTER_LINEAR)

        # Compute temporal PSNR (difference in motion-compensated prediction)
        mse = np.mean((warped_orig - warped_comp) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10((255 ** 2) / mse)
        psnr_values.append(psnr)

    avg_psnr = np.mean(psnr_values)
    return avg_psnr

def compute_temporal_SSIM(orig_frames, comp_frames):
    ssim_values = []
    num_frames = min(len(orig_frames), len(comp_frames))

    for i in range(1, num_frames):
        orig_prev = cv2.cvtColor(orig_frames[i-1], cv2.COLOR_BGR2GRAY)
        orig_curr = cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2GRAY)
        comp_prev = cv2.cvtColor(comp_frames[i-1], cv2.COLOR_BGR2GRAY)
        comp_curr = cv2.cvtColor(comp_frames[i], cv2.COLOR_BGR2GRAY)

        # Motion estimation (optical flow)
        flow_orig = cv2.calcOpticalFlowFarneback(orig_prev, orig_curr, None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow_comp = cv2.calcOpticalFlowFarneback(comp_prev, comp_curr, None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        h, w = orig_curr.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow_orig[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_orig[..., 1]).astype(np.float32)
        warped_orig = cv2.remap(orig_prev, map_x, map_y, cv2.INTER_LINEAR)

        map_x = (grid_x + flow_comp[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_comp[..., 1]).astype(np.float32)
        warped_comp = cv2.remap(comp_prev, map_x, map_y, cv2.INTER_LINEAR)

        # Compute temporal SSIM (difference in motion-compensated prediction)
        ssim_index = ssim(warped_orig, warped_comp, data_range=255)
        ssim_values.append(ssim_index)

    avg_ssim = np.mean(ssim_values)
    return avg_ssim

def MS_SSIM(orig_frames, comp_frames):
    ms_ssim_module = MS_SSIM(data_range=len(orig_frames), size_average=True, channel=3)
    ms_ssim_value = ms_ssim_module(
        torch.from_numpy(orig_frames).permute(0, 3, 1, 2).float(),
        torch.from_numpy(comp_frames).permute(0, 3, 1, 2).float(),
    )
    return ms_ssim_value.item()

if __name__ == "__main__":
    # Example usage
    orig_video_path = "videos/UVG/Jockey_1920x1080_120fps_420_8bit_YUV.y4m"
    comp_video_path = "compressed_videos/UVG/h264/1/Jockey_1920x1080_120fps_420_8bit_YUV.mkv"

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

    temporal_psnr = compute_temporal_psnr(orig_frames, comp_frames)
    temporal_ssim = compute_temporal_SSIM(orig_frames, comp_frames)
    ms_ssim_value = MS_SSIM(np.array(orig_frames), np.array(comp_frames))

    print(f"Temporal PSNR: {temporal_psnr}")
    print(f"Temporal SSIM: {temporal_ssim}")
    print(f"MS-SSIM: {ms_ssim_value}")