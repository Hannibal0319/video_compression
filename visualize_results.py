import matplotlib.pyplot as plt
import numpy as np
import os
import json

datasets = ["UVG", "HEVC_CLASS_B"]
levels = [1, 2, 3]
codecs = ["h264", "hevc", "vp9"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_"
}

def visualize_results_by_codec_psnr(output_dir="visualizations"):
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for codec in codecs:
            for level in levels:
                psnr_list = []
                ssim_list = []
                bpp_list = []
                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)
                
                for video_result in video_results.items():

                    psnr_list.append(video_result['psnr'])
                    ssim_list.append(video_result['ssim'])
                    bpp_list.append(int(level)*1000)
                
                plt.scatter(
                    [res['bpp'] for res in video_results],
                    psnr_list,
                    label=f"{codec.upper()} Level {level}",
                    alpha=0.7
                )
        
        plt.title(f"Rate-Distortion Curve for {dataset}")
        plt.xlabel("Bits per Pixel (bpp)")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve.png"))
        plt.close()


if __name__ == "__main__":
    visualize_results_by_codec_psnr()