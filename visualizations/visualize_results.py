import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import seaborn as sns

datasets = ["UVG"]
levels = [1,1.5, 2, 2.5, 3,4,8]
codecs = ["h264", "hevc", "vp9"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_"
}

def visualize_results_by_codec(output_dir="visualizations"):
    
    for dataset in datasets:
        plt.figure(figsize=(8, 8))
        plt.suptitle(f"Rate-Distortion Curve for {dataset}")

        for codec in codecs:
            metrics_for_codec = {}
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                # average psnr, ssim, vmaf over videos plotted as line with respect to bpp 
                metrics_for_codec[level] = {"psnr": [],
                    "ssim": [],
                    "vmaf": []}
                for video_name, video_data in video_results.items():
                    metrics_for_codec[level]["psnr"].append(video_data["psnr"])
                    metrics_for_codec[level]["ssim"].append(video_data["ssim"])
                    metrics_for_codec[level]["vmaf"].append(video_data["vmaf"])
            # average over all videos for each level
            bpp = []
            psn = []
            ssim = []
            vmaf = []
            for level in levels:
                avg_psnr = np.mean(metrics_for_codec[level]["psnr"])
                avg_ssim = np.mean(metrics_for_codec[level]["ssim"])
                avg_vmaf = np.mean(metrics_for_codec[level]["vmaf"])
                bpp_value = 0.1 * level  # assuming bpp increases linearly with level for simplicity
                bpp.append(bpp_value)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)

            plt.subplot(3, 1, 1)
            plt.plot(bpp, psn, marker='o', label=codec.upper())
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(bpp, ssim, marker='o', label=codec.upper())
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(bpp, vmaf, marker='o', label=codec.upper())
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
        
        plt.xlabel("Bits per Pixel (bpp)")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_codec.png"))
        plt.close()

def visualize_results_by_level(output_dir="visualizations"):
    for dataset in datasets:
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Rate-Distortion Curve for {dataset}")

        for level in levels:
            metrics_for_level = {}
            for codec in codecs:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                # average psnr, ssim, vmaf over videos plotted as line with respect to bpp 
                metrics_for_level[codec] = {"psnr": [],
                    "ssim": [],
                    "vmaf": []}
                for video_name, video_data in video_results.items():
                    metrics_for_level[codec]["psnr"].append(video_data["psnr"])
                    metrics_for_level[codec]["ssim"].append(video_data["ssim"])
                    metrics_for_level[codec]["vmaf"].append(video_data["vmaf"])
            # average over all videos for each codec
            codec_list = []
            psn = []
            ssim = []
            vmaf = []
            for codec in codecs:
                avg_psnr = np.mean(metrics_for_level[codec]["psnr"])
                avg_ssim = np.mean(metrics_for_level[codec]["ssim"])
                avg_vmaf = np.mean(metrics_for_level[codec]["vmaf"])
                codec_lower = codec.lower()
                codec_list.append(codec_lower)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)

            plt.subplot(3, 1, 1)
            plt.plot(codec_list, psn, marker='o', label=f"Level {level}")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(codec_list, ssim, marker='o', label=f"Level {level}")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(codec_list, vmaf, marker='o', label=f"Level {level}")
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
        
        plt.xlabel("Codecs")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_level.png"))
        plt.close()

def visualize_results_by_video(output_dir="visualizations"):
    for dataset in datasets:
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Rate-Distortion Curve per Video for {dataset}")
        for level in levels:
            results_per_video = {}
            for codec in codecs:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    if video_name not in results_per_video:
                        results_per_video[video_name] = {}
                    results_per_video[video_name][codec] = {
                        "bpp": level * 0.1,  # assuming bpp increases linearly with level for simplicity
                        "psnr": video_data["psnr"],
                        "ssim": video_data["ssim"],
                        "vmaf": video_data["vmaf"]
                    }
            
            for video_name, codec_data in results_per_video.items():
                # Plot each video on one plot per level across codecs
                if codec_data.keys() != set(codecs):
                    continue  # skip if any codec data is missing
                bpp = [codec_data[codec]["bpp"] for codec in codecs]
                psn = [codec_data[codec]["psnr"] for codec in codecs]
                ssim = [codec_data[codec]["ssim"] for codec in codecs]
                vmaf = [codec_data[codec]["vmaf"] for codec in codecs]
                plt.subplot(3, 1, 1)
                plt.bar(bpp, psn, label=f"{video_name} Level {level}")
                plt.ylabel("PSNR (dB)")
                plt.grid(True)
                plt.legend()
                plt.subplot(3, 1, 2)
                plt.bar(bpp, ssim, label=f"{video_name} Level {level}")
                plt.ylabel("SSIM")
                plt.grid(True)
                plt.legend()
                plt.subplot(3, 1, 3)
                plt.bar(bpp, vmaf, label=f"{video_name} Level {level}")
                plt.ylabel("VMAF")
                plt.grid(True)
                plt.legend()
        
        plt.xlabel("Bits per Pixel (bpp)")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_video.png"))
        plt.close()

def table_of_results_by_codec():
    for dataset in datasets:
        records = []
        for codec in codecs:
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    records.append({
                        "Dataset": dataset,
                        "Video": video_name.split('_')[0],
                        "Codec": codec,
                        "Level": level,
                        "PSNR": round(video_data["psnr"], 2),
                        "SSIM": round(video_data["ssim"], 4),
                        "VMAF": round(video_data["vmaf"], 2)
                    })
        df = pd.DataFrame.from_records(records)
        # plot table using pandas as image
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        plt.savefig(os.path.join("visualizations", f"{dataset}_results_table_by_codec.png"))
        plt.close()

def visualize_result_by_video_violin_plots(output_dir="visualizations"):
    for dataset in datasets:
        records = []
        for codec in codecs:
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    records.append({
                        "Dataset": dataset,
                        "Video": video_name.split('_')[0],
                        "Codec": codec,
                        "Level": level,
                        "PSNR": video_data["psnr"],
                        "SSIM": video_data["ssim"],
                        "VMAF": video_data["vmaf"]
                    })
        
        for metric in ["PSNR", "SSIM", "VMAF"]:
            df = pd.DataFrame.from_records(records)
            plt.figure(figsize=(8, 4))
            plt.suptitle(f"Distribution of {metric} for {dataset}")
            sns.violinplot(x="Video", y=metric, data=df)
            plt.xlabel("Videos")
            

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset}_violin_plots_{metric}.png"))
            plt.close()


def visualize_results_multi_metric_radar(output_dir="visualizations"):
    for dataset in datasets:
        results_summary = {}
        for codec in codecs:
            results_summary[codec] = {"PSNR": [], "SSIM": [], "VMAF": []}
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    results_summary[codec]["PSNR"].append(video_data["psnr"])
                    results_summary[codec]["SSIM"].append(video_data["ssim"])
                    results_summary[codec]["VMAF"].append(video_data["vmaf"])
        
        # Average metrics per codec (original values)
        avg_metrics = {}
        for codec, metrics in results_summary.items():
            avg_metrics[codec] = {
                "PSNR": np.mean(metrics["PSNR"]),
                "SSIM": np.mean(metrics["SSIM"]),
                "VMAF": np.mean(metrics["VMAF"])
            }
        
        # Normalize each metric to a common 0-100 scale so SSIM is visible on the radar
        metric_ranges = {
            "PSNR": (0.0, 50.0),   # expected PSNR range
            "SSIM": (0.0, 1.0),    # SSIM range
            "VMAF": (0.0, 100.0)   # VMAF range
        }
        def normalize_to_100(value, metric):
            lo, hi = metric_ranges[metric]
            if hi == lo:
                return 0.0
            return 100.0 * (np.clip(value, lo, hi) - lo) / (hi - lo)

        labels = ["PSNR", "SSIM", "VMAF"]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for codec, metrics in avg_metrics.items():
            values_orig = [metrics[label] for label in labels]
            values_norm = [normalize_to_100(v, label) for v, label in zip(values_orig, labels)]
            values_norm += values_norm[:1]
            ax.plot(angles, values_norm, label=f"{codec.upper()} ({', '.join([f'{l}: {round(v,3)}' for l,v in zip(labels, values_orig)])})")
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
    visualize_results_by_codec()
    visualize_results_by_level()
    #visualize_results_by_video()
    
    #table_of_results_by_codec()
    #visualize_result_by_video_violin_plots()
    #visualize_results_multi_metric_radar()
    pass