import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import seaborn as sns

datasets = ["UVG","HEVC_CLASS_B"]
levels = [1,1.5, 2, 2.5, 3,4,8]
codecs = ["h264", "hevc", "vp9"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_"
}



def visualize_result_by_video_violin_plots(output_dir="visualizations/plots_by_metric"):
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
                        "PSNR": video_data["psnr"] if "psnr" in video_data else None,
                        "SSIM": video_data["ssim"] if "ssim" in video_data else None,
                        "VMAF": video_data["vmaf"] if "vmaf" in video_data else None,
                        "tPSNR": video_data["tpsnr"] if "tpsnr" in video_data else None,
                        "tSSIM": video_data["tssim"] if "tssim" in video_data else None,
                        "FVD": video_data["fvd"] if "fvd" in video_data else None,
                        "Movie Index": video_data["movie_index"] if "movie_index" in video_data else None,
                        "ST-RRED": video_data["st_rred"] if "st_rred" in video_data else None
                    })
        
        for metric in ["PSNR", "SSIM", "VMAF" , "tPSNR", "tSSIM", "FVD", "Movie Index", "ST-RRED"]:
            df = pd.DataFrame.from_records(records)
            plt.figure(figsize=(8, 4))
            plt.suptitle(f"Distribution of {metric} for {dataset}")
            sns.violinplot(x="Video", y=metric, data=df)
            plt.xlabel("Videos")
            

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset}_violin_plots_{metric}.png"))
            plt.close()




if __name__ == "__main__":
    visualize_result_by_video_violin_plots()
    pass