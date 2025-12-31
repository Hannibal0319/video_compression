import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# List of files (simulated based on the context provided, but in real env I'd list dir)
# The user uploaded files are in the current directory.
files = [
    "bd_scores_HEVC_CLASS_B_h264.json",
    "bd_scores_UVG_hevc.json",
    "bd_scores_UVG_h264.json",
    "bd_scores_HEVC_CLASS_B_vp9.json",
    "bd_scores_UVG_av1.json",
    "bd_scores_UVG_vp9.json",
    "bd_scores_HEVC_CLASS_B_hevc.json",
    "bd_scores_HEVC_CLASS_B_av1.json",
    "bd_scores_BVI-HD_h264.json",
    "bd_scores_BVI-HD_hevc.json",
    "bd_scores_BVI-HD_vp9.json",
    "bd_scores_BVI-HD_av1.json"
]

data_list = []

for filename in files:
    # Parsing filename for metadata
    # Expected format: bd_scores_{Dataset}_{Anchor}.json
    # But one dataset is "HEVC_CLASS_B" (with underscores) and one is "UVG".
    # And anchors are h264, hevc, vp9, av1.
    
    base_name = filename.replace("bd_scores_", "").replace(".json", "")
    
    # Identify anchor (last part after last underscore)
    parts = base_name.split('_')
    anchor = parts[-1]
    
    # Identify dataset (everything before the anchor)
    dataset = "_".join(parts[:-1])
    
    filename = "results/bd_scores/" + filename
    with open(filename, 'r') as f:
        content = json.load(f)
        
    for tested_codec, sequences in content.items():
        for seq_name, metrics in sequences.items():
            row = {
                'Dataset': dataset,
                'Anchor': anchor,
                'Tested_Codec': tested_codec,
                'Sequence': seq_name
            }
            # Add all metrics
            row.update(metrics)
            data_list.append(row)

df = pd.DataFrame(data_list)
print(df.head())
print(df['Dataset'].unique())
print(df.columns)

import numpy as np

# Re-loading/Processing to handle aggregations
# We already have 'df'.

metrics_to_plot = [m for m in ['psnr', 'ssim', 'vmaf', 'fvd', 'tssim', 'tpsnr', 'st_rred', 'movie_index'] if m in df.columns]

# Check for infinity or nan in these metrics and clean
df_clean = df.replace([np.inf, -np.inf], np.nan)

# Group by Dataset, Anchor, Tested_Codec
# We take the mean across sequences
df_grouped = df_clean.groupby(['Dataset', 'Anchor', 'Tested_Codec'])[metrics_to_plot].mean().reset_index()

datasets = df_grouped['Dataset'].unique()
codecs = ['h264', 'hevc', 'vp9', 'av1'] # Expected ordering

for dataset in datasets:
    # Filter for dataset
    data_ds = df_grouped[df_grouped['Dataset'] == dataset]
    
    # Prepare figure
    num_metrics = len(metrics_to_plot)
    cols = 4
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f'Average BD-Rate Scores - {dataset}\n(Negative is Better)', fontsize=20)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Pivot to create matrix: Index=Anchor, Col=Tested_Codec
        pivot_table = data_ds.pivot(index='Anchor', columns='Tested_Codec', values=metric)
        
        # Reindex to ensure all codecs are present and in order
        pivot_table = pivot_table.reindex(index=codecs, columns=codecs)
        
        # Fill diagonal with 0 (Anchor vs itself)
        for codec in codecs:
            pivot_table.loc[codec, codec] = 0.0
            
        # Plot heatmap
        # Use RdYlGn_r: Green (low/negative) to Red (high/positive)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn_r", center=0, ax=ax, cbar=True)
        ax.set_title(metric.upper())
        ax.set_ylabel("Anchor Codec")
        ax.set_xlabel("Tested Codec")
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("visualizations/bd_scores_plots/bd_rate_scores_" + dataset + ".png")
    print(f"Saved plot for dataset: {dataset}")
    plt.close()

print("Plots created.")