# This script computes BD (Bjøntegaard Delta) scores for video compression performance comparison.
import os
import json
from pathlib import Path
import subprocess
import numpy as np
import bjontegaard as bd

datasets = ["UVG", "HEVC_CLASS_B", "BVI-HD"]
levels = [1, 1.5, 2, 2.5, 3, 4, 8]
compute_metrics =["psnr","ssim","vmaf","fvd","st_rred","tssim","tpsnr","movie_index"]
codecs = ["h264", "hevc", "av1", "vp9"]
    
def compute_bd_rate(rate1, metric1, rate2, metric2):
    """
    Computes the Bjøntegaard Delta Rate between two rate-distortion curves.
    rate1, metric1: Lists of rates and corresponding quality metrics for the first curve.
    rate2, metric2: Lists of rates and corresponding quality metrics for the second curve.
    Returns the BD-Rate percentage difference.
    """
    bd_rate = bd.bd_rate(rate1, metric1, rate2, metric2,method='akima',min_overlap=0)
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
                        if metric in ["fvd", "movie_index","st_rred"]:
                            # For these metrics, lower is better, so we invert them for BD-rate calculation
                            metrics[metric] = -metrics[metric]
                        else:
                            metrics[metric] = metrics[metric]
                        all_data[dataset][codec][video][metric].append(metrics[metric])
    
    print("Data loaded. Computing BD scores...")
    #print("data with only strred:", {ds: {c: {v: d[c][v]['st_rred'] for v in d[c]} for c in d} for ds, d in all_data.items()})
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

                    # Skip if there are NaN values, which can cause errors.
                    if np.isnan(codec_metrics).any() or np.isnan(ref_metrics).any():
                        print(f"Skipping BD-rate for {dataset}/{codec}/{video}/{metric} due to NaN values.")
                        continue
                    
                    # Enforce monotonicity: ensure that metric values are strictly increasing
                    codec_metrics = np.maximum.accumulate(codec_metrics)
                    ref_metrics = np.maximum.accumulate(ref_metrics)

                    # Ensure the sequence is strictly increasing by adding a small epsilon to duplicates
                    for i in range(1, len(codec_metrics)):
                        if codec_metrics[i] <= codec_metrics[i-1]:
                            codec_metrics[i] = codec_metrics[i-1] + 1e-9
                    for i in range(1, len(ref_metrics)):
                        if ref_metrics[i] <= ref_metrics[i-1]:
                            ref_metrics[i] = ref_metrics[i-1] + 1e-9

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

