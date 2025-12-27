#compute bending energy of curve drawn by the video in frame space
import numpy as np
import tqdm

def bending_energy_1080x1920x3(curve):
    """
    Compute the bending energy of a curve in 1080x1920x3 space.

    Parameters:
    curve (np.ndarray): An Nx3 array representing the curve points in 1080x1920x3 space.

    Returns:
    float: The bending energy of the curve.
    """
    n_points = curve.shape[0]
    if n_points < 3:
        return 0.0  # Not enough points to compute bending energy

    energy = 0.0
    for i in tqdm.tqdm(range(1, n_points - 1)):
        p_prev = curve[i - 1]
        p_curr = curve[i]
        p_next = curve[i + 1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            continue  # Avoid division by zero

        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical stability
        theta = np.arccos(cos_theta)

        energy += theta ** 2

    return energy

import cv2


def load_curve_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    curve = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print("Frame shape:", frame.shape)
        curve.append(frame.flatten())
    
    video_capture.release()
    return np.array(curve)

def compute_bending_energy_from_video(video_path):
    curve = load_curve_from_video(video_path)
    print("Curve shape:", curve.shape)
    energy = bending_energy_1080x1920x3(curve)
    
    normalized_energy = energy / int(curve.shape[0])
    return normalized_energy

def compute_bending_energy_from_videos(video_paths):
    energies = {}
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        energy = compute_bending_energy_from_video(video_path)
        print(f"Bending energy for {video_path}: {energy}")
        energies[video_path] = energy
    return energies

import json
import os

def compute_energy_for_dataset(datasets):
    for dataset_name in datasets:
        video_paths = []
        dataset_path = os.path.join("videos", dataset_name)
        for filename in os.listdir(dataset_path):
            if filename.endswith(".mp4") or filename.endswith(".y4m"):
                video_paths.append(os.path.join(dataset_path, filename))
        energies = compute_bending_energy_from_videos(video_paths)
        with open(f"{dataset_name}_bending_energies.json", "w") as f:
            json.dump(energies, f, indent=4)
    

if __name__ == "__main__":
    datasets = ["HEVC_CLASS_B"]
    compute_energy_for_dataset(datasets)