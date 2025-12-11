import numpy as np
import json
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

datasets = ['UVG', 'HEVC_CLASS_B']
levels = [1, 1.5, 2, 2.5, 3, 4, 8]
compute_metrics = ['psnr', 'ssim', 'vmaf', 'fvd', 'tssim', 'tpsnr', 'movie_index', 'st_rred']
codecs = ['h264', 'hevc', 'av1', 'vp9']

def visualize_bd_scores(input_dir="results/bd_scores", output_dir="visualizations/bd_scores_plots"):
    """
    Generates and saves line plots for BD scores from JSON files.

    For each JSON file in the input directory, this function creates a set of line plots,
    one for each metric. Each plot displays the average BD-rate for different codecs
    relative to a reference codec, based on the data in the file.

    Args:
        input_dir (str): The directory containing the BD score JSON files.
        output_dir (str): The directory where the generated plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.startswith("bd_scores_") and filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            
            # Extract dataset and reference codec from filename
            parts = filename.replace("bd_scores_", "").replace(".json", "").split('_')
            dataset = parts[0]
            reference_codec = '_'.join(parts[1:]) # Handle cases like HEVC_CLASS_B

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert nested dictionary to a more usable format
            records = []
            for codec, videos in data.items():
                for video, metrics in videos.items():
                    for metric, value in metrics.items():
                        print(video)
                        # Replace infinity with a large number for plotting or skip
                        if value == float('inf'):
                            value = np.nan  # Use NaN to skip plotting this point
                        records.append({
                            "codec": codec,
                            "video": video,
                            "metric": metric,
                            "bd_rate": value,

                        })
            
            if not records:
                print(f"No data to plot for {filename}")
                continue

            df = pd.DataFrame(records)

            # Get unique metrics
            metrics = df['metric'].unique()
            
            for metric in metrics:
                plt.figure(figsize=(12, 7))
                
                metric_df = df[df['metric'] == metric].dropna(subset=['bd_rate'])
                
                if metric_df.empty:
                    plt.close()
                    continue

                # Use pivot_table to get average BD-rate per codec
                pivot_df = metric_df.pivot_table(index='', values='bd_rate', aggfunc='mean')
                
                if pivot_df.empty:
                    plt.close()
                    continue

                pivot_df = pivot_df.sort_index()

                plt.plot(pivot_df.index, pivot_df['bd_rate'], marker='o', linestyle='-')

                plt.title(f'Average BD-Rate for {metric.upper()} ({dataset} vs {reference_codec.upper()})')
                plt.xlabel('Codec')
                plt.ylabel('Average BD-Rate (%)')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.axhline(0, color='red', linewidth=0.8, linestyle='--')
                
                # Save the plot
                plot_filename = f"{dataset}_{reference_codec}_{metric}_bd_rate.png"
                output_path = os.path.join(output_dir, plot_filename)
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()

    print(f"BD score visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_bd_scores()
