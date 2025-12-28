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
        with open(f"results/energy/{dataset_name}_bending_energies.json", "w") as f:
            json.dump(energies, f, indent=4)
    

def load_TI_groups():
    """Return TI group lookup per dataset using fixed TI thresholds."""
    group_bounds = [11.495521545410156, 17.721065521240234, 29.252790451049805]
    datasets = ["HEVC_CLASS_B", "UVG", "BVI-HD"]

    ti_groups = {}
    for dataset_name in datasets:
        ti_path = os.path.join("results", f"eval_metrics_{dataset_name}_TI.json")
        with open(ti_path, "r") as f:
            ti_data = json.load(f)

        video_groups = {}
        for video_name, ti_value in ti_data.items():
            if ti_value <= group_bounds[0]:
                group = 1
            elif ti_value <= group_bounds[1]:
                group = 2
            elif ti_value <= group_bounds[2]:
                group = 3
            else:
                group = 4
            video_groups[video_name] = group

        ti_groups[dataset_name] = video_groups

    return ti_groups

def enery_by_TI_group():
    datasets = ["HEVC_CLASS_B","UVG","BVI-HD"]
    TI_groups = load_TI_groups()
    for dataset_name in datasets:
        with open(f"results/energy/{dataset_name}_bending_energies.json", "r") as f:
            energies = json.load(f)
        TI_group_energies = {}
        for video_path, energy in energies.items():
            video_name = os.path.basename(video_path)
            TI_group = TI_groups[dataset_name].get(video_name, "Unknown")
            if TI_group not in TI_group_energies:
                TI_group_energies[TI_group] = []
            TI_group_energies[TI_group].append(energy)
        avg_TI_group_energies = {group: np.mean(vals) for group, vals in TI_group_energies.items()}
        with open(f"results/energy/{dataset_name}_bending_energies_by_TI_group.json", "w") as f:
            json.dump(avg_TI_group_energies, f, indent=4)
    
    #make avg across datasets
    combined_TI_group_energies = {}
    for dataset_name in datasets:
        with open(f"{dataset_name}_bending_energies_by_TI_group.json", "r") as f:
            dataset_TI_group_energies = json.load(f)
        for group, energy in dataset_TI_group_energies.items():
            if group not in combined_TI_group_energies:
                combined_TI_group_energies[group] = []
            combined_TI_group_energies[group].append(energy)
    
    avg_combined_TI_group_energies = {group: np.mean(vals) for group, vals in combined_TI_group_energies.items()}
    print("Average Bending Energy by TI Group across Datasets sorted by energy:")
    for group, energy in sorted(avg_combined_TI_group_energies.items(), key=lambda item: item[1]):
        print(f"TI Group {group}: {energy}")
    
    print()

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def calculate_average_cosine_similarity(video_path):
    curve = load_curve_from_video(video_path)
    n_points = curve.shape[0]
    if n_points < 2:
        return 0.0  # Not enough points to compute cosine similarity

    total_similarity = 0.0
    count = 0
    for i in range(n_points - 1):
        v1 = curve[i]
        v2 = curve[i + 1]
        similarity = cosine_similarity(v1, v2)
        total_similarity += similarity
        count += 1

    average_similarity = total_similarity / count if count > 0 else 0.0
    return average_similarity

def compute_cosine_similarities_from_videos(video_paths):
    similarities = {}
    for video_path in video_paths:
        print(f"Processing video for cosine similarity: {video_path}")
        similarity = calculate_average_cosine_similarity(video_path)
        print(f"Average Cosine Similarity for {video_path}: {similarity}")
        similarities[video_path] = similarity
    return similarities

def compute_cosine_similarity_for_dataset(datasets):
    for dataset_name in datasets:
        video_paths = []
        dataset_path = os.path.join("videos", dataset_name)
        for filename in os.listdir(dataset_path):
            if filename.endswith(".mp4") or filename.endswith(".y4m"):
                video_paths.append(os.path.join(dataset_path, filename))
        similarities = compute_cosine_similarities_from_videos(video_paths)
        with open(f"results/smoothness_index/{dataset_name}_cosine_similarities.json", "w") as f:
            json.dump(similarities, f, indent=4)

def cosine_per_TI_group(datasets):
    TI_groups = load_TI_groups()
    for dataset_name in datasets:
        with open(f"results/smoothness_index/{dataset_name}_cosine_similarities.json", "r") as f:
            similarities = json.load(f)
        TI_group_similarities = {}
        for video_path, similarity in similarities.items():
            video_name = os.path.basename(video_path)
            TI_group = TI_groups[dataset_name].get(video_name, "Unknown")
            if TI_group not in TI_group_similarities:
                TI_group_similarities[TI_group] = []
            TI_group_similarities[TI_group].append(similarity)
        avg_TI_group_similarities = {group: np.mean(vals) for group, vals in TI_group_similarities.items()}
        with open(f"results/smoothness_index/{dataset_name}_cosine_similarities_by_TI_group.json", "w") as f:
            json.dump(avg_TI_group_similarities, f, indent=4)
    
    #make avg across datasets
    combined_TI_group_similarities = {}
    for dataset_name in datasets:
        with open(f"results/smoothness_index/{dataset_name}_cosine_similarities_by_TI_group.json", "r") as f:
            dataset_TI_group_similarities = json.load(f)
        for group, similarity in dataset_TI_group_similarities.items():
            if group not in combined_TI_group_similarities:
                combined_TI_group_similarities[group] = []
            combined_TI_group_similarities[group].append(similarity)
    
    avg_combined_TI_group_similarities = {group: np.mean(vals) for group, vals in combined_TI_group_similarities.items()}
    print("Average Cosine Similarity by TI Group across Datasets sorted by similarity:")
    for group, similarity in sorted(avg_combined_TI_group_similarities.items(), key=lambda item: item[1], reverse=True):
        print(f"TI Group {group}: {similarity}")
    
    print()

def SVD_entropy(curve):
    """
    Compute the SVD entropy of a curve.

    Parameters:
    curve (np.ndarray): An NxM array representing the curve points.

    Returns:
    float: The SVD entropy of the curve.
    """
    U, S, Vt = np.linalg.svd(curve - np.mean(curve, axis=0), full_matrices=False)
    S_normalized = S / np.sum(S)
    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))  # Adding small value for numerical stability
    return entropy

import torch
from typing import Optional


def _svd_entropy_from_tensor(curve_tensor: torch.Tensor) -> float:
    """Compute entropy from a centered curve tensor without building gradients."""
    curve_centered = curve_tensor - torch.mean(curve_tensor, dim=0, keepdim=True)
    # svdvals avoids materializing U/V and saves memory compared to full svd
    singular_values = torch.linalg.svdvals(curve_centered)
    singular_values = singular_values / torch.sum(singular_values)
    entropy = -torch.sum(singular_values * torch.log(singular_values + 1e-10))
    return entropy.item()


def SVD_entropy_torch(curve, device: Optional[str] = None):
    """
    Compute the SVD entropy of a curve using PyTorch with GPU-safe fallback.

    Parameters:
    curve (np.ndarray or torch.Tensor): An NxM array representing the curve points.
    device (str, optional): "cuda" or "cpu". Defaults to cuda if available.

    Returns:
    float: The SVD entropy of the curve.
    """
    chosen_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curve_tensor = torch.as_tensor(curve, dtype=torch.float32, device=chosen_device)

    with torch.no_grad():
        entropy = _svd_entropy_from_tensor(curve_tensor)
    return entropy
    

def compute_SVD_entropy_from_video(video_path):
    curve = load_curve_from_video(video_path)
    entropy = SVD_entropy_torch(curve)
    return entropy

def compute_SVD_entropy_from_videos(video_paths):
    entropies = {}
    for video_path in video_paths:
        print(f"Processing video for SVD entropy: {video_path}")
        entropy = compute_SVD_entropy_from_video(video_path)
        print(f"SVD Entropy for {video_path}: {entropy}")
        entropies[video_path] = entropy
    return entropies

def compute_SVD_entropy_for_dataset(datasets):
    for dataset_name in datasets:
        video_paths = []
        dataset_path = os.path.join("videos", dataset_name)
        for filename in os.listdir(dataset_path):
            if filename.endswith(".mp4") or filename.endswith(".y4m"):
                video_paths.append(os.path.join(dataset_path, filename))
        entropies = compute_SVD_entropy_from_videos(video_paths)
        with open(f"results/SVD_entropy/{dataset_name}_SVD_entropies.json", "w") as f:
            json.dump(entropies, f, indent=4)

def SVD_entropy_per_TI_group(datasets):
    TI_groups = load_TI_groups()
    for dataset_name in datasets:
        with open(f"results/SVD_entropy/{dataset_name}_SVD_entropies.json", "r") as f:
            entropies = json.load(f)
        TI_group_entropies = {}
        for video_path, entropy in entropies.items():
            video_name = os.path.basename(video_path)
            TI_group = TI_groups[dataset_name].get(video_name, "Unknown")
            if TI_group not in TI_group_entropies:
                TI_group_entropies[TI_group] = []
            TI_group_entropies[TI_group].append(entropy)
        avg_TI_group_entropies = {group: np.mean(vals) for group, vals in TI_group_entropies.items()}
        with open(f"{dataset_name}_SVD_entropies_by_TI_group.json", "w") as f:
            json.dump(avg_TI_group_entropies, f, indent=4)
    
    #make avg across datasets
    combined_TI_group_entropies = {}
    for dataset_name in datasets:
        with open(f"results/SVD_entropy/{dataset_name}_SVD_entropies_by_TI_group.json", "r") as f:
            dataset_TI_group_entropies = json.load(f)
        for group, entropy in dataset_TI_group_entropies.items():
            if group not in combined_TI_group_entropies:
                combined_TI_group_entropies[group] = []
            combined_TI_group_entropies[group].append(entropy)
    
    avg_combined_TI_group_entropies = {group: np.mean(vals) for group, vals in combined_TI_group_entropies.items()}
    print("Average SVD Entropy by TI Group across Datasets sorted by entropy:")
    for group, entropy in sorted(avg_combined_TI_group_entropies.items(), key=lambda item: item[1]):
        print(f"TI Group {group}: {entropy}")
    
    print()

if __name__ == "__main__":
    datasets = ["HEVC_CLASS_B","UVG","BVI-HD"]
    compute_SVD_entropy_for_dataset(datasets)
    SVD_entropy_per_TI_group(datasets)