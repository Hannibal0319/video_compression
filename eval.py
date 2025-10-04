import numpy as np
import cv2

def eval_PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def eval_SSIM(original, compressed):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)

    mu_x = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(compressed, (11, 11), 1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(compressed ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(original * compressed, (11, 11), 1.5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()

def eval_VMAF(original, compressed):
    # call the VMAF library or use subprocess to call the VMAF executable
    # This is a placeholder function; actual implementation would depend on VMAF setup
    return 0.0

def evaluate_video(original_path, compressed_path):
    cap_orig = cv2.VideoCapture(original_path)
    cap_comp = cv2.VideoCapture(compressed_path)

    psnr_values = []
    ssim_values = []

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_comp, frame_comp = cap_comp.read()

        if not ret_orig or not ret_comp:
            break

        psnr = eval_PSNR(frame_orig, frame_comp)
        ssim = eval_SSIM(frame_orig, frame_comp)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    cap_orig.release()
    cap_comp.release()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim