# This script computes BD (Bjøntegaard Delta) scores for video compression performance comparison.
import os
import json
from pathlib import Path
import subprocess
import numpy as np
import re
import pandas as pd

datasets = ["UVG", "HEVC_CLASS_B"]
levels = [1, 1.5, 2, 2.5, 3, 4, 8]
compute_metrics =["psnr","ssim","vmaf","fvd","tssim","tpsnr","movie_index","st_rred"]
codecs = ["h264", "hevc", "av1", "vp9"]

def run_command(cmd):
    """Runs a command and returns its output."""
    try:
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        return e.output
    
def compute_bd_rate(rate1, metric1, rate2, metric2):
    """
    Computes the Bjøntegaard Delta Rate between two rate-distortion curves.
    rate1, metric1: Lists of rates and corresponding quality metrics for the first curve.
    rate2, metric2: Lists of rates and corresponding quality metrics for the second curve.
    Returns the BD-Rate percentage difference.
    """
    # Convert to numpy arrays for easier manipulation
    rate1 = np.array(rate1)
    metric1 = np.array(metric1)
    rate2 = np.array(rate2)
    metric2 = np.array(metric2)

    # Fit cubic polynomials to the log-rate vs metric data
    p1 = np.polyfit(metric1, np.log(rate1), 3)
    p2 = np.polyfit(metric2, np.log(rate2), 3)

    # Define the integration limits
    min_metric = max(min(metric1), min(metric2))
    max_metric = min(max(metric1), max(metric2))

    # Integrate the fitted polynomials over the common metric range
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)

    int1 = np.polyval(p_int1, max_metric) - np.polyval(p_int1, min_metric)
    int2 = np.polyval(p_int2, max_metric) - np.polyval(p_int2, min_metric)

    # Compute the average difference in log-rate
    # Handle case where max_metric is very close to min_metric
    if np.isclose(max_metric, min_metric):
        avg_diff = 0
    else:
        avg_diff = (int2 - int1) / (max_metric - min_metric)

    # Handle potential overflow
    if avg_diff > 700:  # np.exp(709) is near max float64, 700 is a safe cap
        return 1000000000.0  # Return a large number to indicate overflow

    # Convert back from log-rate to percentage
    bd_rate = (np.exp(avg_diff) - 1) * 100

    return bd_rate
    
def load_rate_distortion_data(json_path):
    """Loads rate-distortion data from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rates = data['rates']
    metrics = data['metrics']
    return rates, metrics

def process_bd_scores(input_dir, output_dir="results/bd_scores",reference_codec="h264"):
    """
    Processes evaluation metric files to compute and store BD scores.
    It scans for JSON files named like 'eval_metrics_{dataset}_{codec}_level{level}.json',
    groups data by dataset, video, and codec, then computes BD-rates for each metric
    against a reference codec (h264), saving the results in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(input_dir)
    
    # Structure to hold all parsed data: {dataset: {codec: {video: {metric: [values], "rates": [values]}}}}
    all_data = {}
    for dataset in datasets:
        all_data[dataset] = {}
        for codec in codecs:
            all_data[dataset][codec] = {}
            for level in levels:
                json_path = results_path / f"eval_metrics_{dataset}_{codec}_level{level}.json"

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for video, metrics in data.items():
                    video = "_".join(video.split("_")[:-1])  # Remove level suffix from video name
                    if video not in all_data[dataset][codec]:
                        all_data[dataset][codec][video] = {'rates': []}
                    for metric in compute_metrics:
                        if metric not in all_data[dataset][codec][video]:
                            all_data[dataset][codec][video][metric] = []
                    all_data[dataset][codec][video]['rates'].append(level * 1000)
                    for metric in compute_metrics:
                        all_data[dataset][codec][video][metric].append(metrics[metric])
    # Compute BD scores
    bd_scores = {}
    for dataset in all_data:
        bd_scores[dataset] = {}
        reference_data = all_data[dataset].get(reference_codec)
        if not reference_data:
            print(f"Reference codec {reference_codec} data not found for dataset {dataset}. Skipping.")
            continue

        for codec, codec_data in all_data[dataset].items():
            if codec == reference_codec:
                continue
            bd_scores[dataset][codec] = {}
            for video, video_data in codec_data.items():
                if video not in reference_data:
                    print(video,reference_data.keys())
                    print(f"Video {video} not found in reference codec data for dataset {dataset}. Skipping.")
                    continue
                bd_scores[dataset][codec][video] = {}
                ref_video_data = reference_data[video]
                
                # Sort by rates before computing BD-rate
                codec_rates = np.array(video_data['rates'])
                ref_rates = np.array(ref_video_data['rates'])

                codec_sort_indices = np.argsort(codec_rates)
                ref_sort_indices = np.argsort(ref_rates)

                sorted_codec_rates = codec_rates[codec_sort_indices]
                sorted_ref_rates = ref_rates[ref_sort_indices]

                for metric in compute_metrics:
                    codec_metrics = np.array(video_data[metric])[codec_sort_indices]
                    ref_metrics = np.array(ref_video_data[metric])[ref_sort_indices]
                    
                    try:
                        bd_rate_val = compute_bd_rate(sorted_ref_rates, ref_metrics, sorted_codec_rates, codec_metrics)
                        bd_scores[dataset][codec][video][metric] = bd_rate_val
                    except Exception as e:
                        print(f"Could not compute BD-rate for {dataset}/{codec}/{video}/{metric}: {e}")

    # Save BD scores to output directory
    for dataset in bd_scores:
        output_path = Path(output_dir) / f"bd_scores_{dataset}_{reference_codec}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bd_scores[dataset], f, indent=4)
    print(f"BD scores saved to {output_dir}")


if __name__ == "__main__":
    process_bd_scores("results", reference_codec="h264")
    process_bd_scores("results", reference_codec="hevc")
    process_bd_scores("results", reference_codec="vp9")
    process_bd_scores("results", reference_codec="av1")

