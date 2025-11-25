import os
import json

import numpy as np
import matplotlib.pyplot as plt

datasets = ["UVG","HEVC_CLASS_B"]
levels = [1,1.5,2,2.5,3,4,8]
codecs = ["h264", "hevc", "vp9"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_"
}



def normalize_metric(value, metric):
    """
    Normalize a metric value to a 0-100 scale based on expected ranges.
    Adjust these ranges based on your specific data characteristics.
    """
    metric_ranges = {
        "psnr": (0.0, 50.0),   # expected PSNR range
        "ssim": (0.0, 1.0),    # SSIM range
        "vmaf": (0.0, 100.0),   # VMAF range
        "tpsnr": (0.0, 50.0),  # expected tPSNR range
        "tssim": (0.0, 1.0),    # tSSIM range
        "fvd": (0.0, 1500.0),    # FVD range
        "movie_index": (0.0, 1.0),  # Movie Index range (example)
        "st_rred": (0.0, 1.0)    # ST-RRED range (example)
    }
    lo, hi = metric_ranges[metric]
    ret = 100.0 * (np.clip(value, lo, hi) - lo) / (hi - lo)
    if metric in ["fvd"]:  # for metrics where lower is better
        ret = 100.0 - ret
    return ret

def get_video_name_stem(video_filename):
    return "_".join(video_filename.split('_')[0:-1])

def visualize_results_by_video_radar_plot(output_dir="visualizations/plots_by_video"):
    for dataset in datasets:

        for level in levels:
            results_per_video = {}
            for codec in codecs:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)
                    #print(f"Loaded results for {dataset} codec {codec} level {level}")
                    #print(f"Number of videos: {len(video_results)}")

                for video_fullname, video_data in video_results.items():
                    video_name = get_video_name_stem(video_fullname)
                    if video_name not in results_per_video:
                        results_per_video[video_name] = {}
                    results_per_video[video_name][codec] = {
                        "bpp": level * 0.1,  # assuming bpp increases linearly with level for simplicity
                        "psnr": normalize_metric(video_data["psnr"], "psnr") if "psnr" in video_data else 0,
                        "ssim": normalize_metric(video_data["ssim"], "ssim") if "ssim" in video_data else 0,
                        "vmaf": normalize_metric(video_data["vmaf"], "vmaf") if "vmaf" in video_data else 0,
                        "tpsnr": normalize_metric(video_data["tpsnr"], "tpsnr") if "tpsnr" in video_data else 0,
                        "tssim": normalize_metric(video_data["tssim"], "tssim") if "tssim" in video_data else 0,
                        "fvd": normalize_metric(video_data["fvd"], "fvd") if "fvd" in video_data else 0,
                        "movie_index": normalize_metric(video_data["movie_index"], "movie_index") if "movie_index" in video_data else 0,
                        "st_rred": normalize_metric(video_data["st_rred"], "st_rred") if "st_rred" in video_data else 0,
                    }
                    #print(results_per_video[video_name][codec])
            
            for video_name in results_per_video.keys():
                plt.figure(figsize=(8, 8))
                plt.suptitle(f"Normalized Radar Plot for {video_name} at {level}kbpp")
                ax = plt.subplot(111, polar=True)
                for codec, values in results_per_video[video_name].items():
                    labels = ["PSNR", "SSIM", "VMAF", "tPSNR", "tSSIM", "FVD", "Movie Index", "ST-RRED"]
                    num_vars = len(labels)
                    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                    angles += angles[:1]
                    metric_values = [
                        values["psnr"],
                        values["ssim"],
                        values["vmaf"],
                        values["tpsnr"],
                        values["tssim"],
                        values["fvd"],
                        values["movie_index"],
                        values["st_rred"]
                    ]
                    metric_values += metric_values[:1]
                    ax.plot(angles, metric_values, label=f"{codec.upper()}")
                    ax.fill(angles, metric_values, alpha=0.25)
                    print(f"Metrics for {codec} on video {video_name} at level {level} in {dataset}:")
                # Show percentage ticks (normalized)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"{dataset}_{video_name}_level{level}_radar_plot.png"))
                plt.close()

def visualize_results_multi_metric_radar_avg_of_videos(output_dir="visualizations"):
    for dataset in datasets:
        results_summary = {}
        for codec in codecs:
            results_summary[codec] = {"PSNR": [], "SSIM": [], "VMAF": [], "tPSNR": [], "tSSIM": [], "FVD": [], "Movie Index": [], "ST-RRED": []}
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    print(f"Processing {dataset} {codec} level {level} video {video_name}")
                    results_summary[codec]["PSNR"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    results_summary[codec]["SSIM"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    results_summary[codec]["VMAF"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    results_summary[codec]["tPSNR"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    results_summary[codec]["tSSIM"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    results_summary[codec]["FVD"].append(video_data["fvd"] if "fvd" in video_data else 0)
                    results_summary[codec]["Movie Index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                    results_summary[codec]["ST-RRED"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
        # Average metrics per codec (original values)
        avg_metrics = {}
        for codec, metrics in results_summary.items():
            avg_metrics[codec] = {
                "PSNR": np.mean(metrics["PSNR"]),
                "SSIM": np.mean(metrics["SSIM"]),
                "VMAF": np.mean(metrics["VMAF"]),
                "tPSNR": np.mean(metrics["tPSNR"]),
                "tSSIM": np.mean(metrics["tSSIM"]),
                "FVD": np.mean(metrics["FVD"]),
                "Movie Index": np.mean(metrics["Movie Index"]),
                "ST-RRED": np.mean(metrics["ST-RRED"])
            }



        labels = ["PSNR", "SSIM", "VMAF", "tPSNR", "tSSIM", "FVD", "Movie Index", "ST-RRED"]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)

        for codec, metrics in avg_metrics.items():
            values_orig = [metrics[label] for label in labels]
            values_norm = [normalize_metric(v, label.lower().replace(" ", "_").replace("-", "_")) for v, label in zip(values_orig, labels)]
            values_norm += values_norm[:1]
            print(f"Metrics for {codec} on {dataset}:")
            print(labels)
            print(f"{codec} average metrics: {values_orig}")
            print(f"{codec} normalized metrics: {values_norm}")
            ax.plot(angles, values_norm, label=f"{codec.upper()}")
            ax.fill(angles, values_norm, alpha=0.25)

        # Show percentage ticks (normalized)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title(f"Normalized Average Metrics Radar Chart for {dataset}\n(metrics normalized to 0-100 for visibility)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{dataset}_radar_chart_metrics.png"))
        plt.close()

if __name__ == "__main__":
    #visualize_results_by_video_radar_plot()
    visualize_results_multi_metric_radar_avg_of_videos()
    pass