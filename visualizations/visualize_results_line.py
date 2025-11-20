from matplotlib import pyplot as plt
import numpy as np
import os
import json

datasets = ["UVG", "HEVC_CLASS_B"]
levels = [1,1.5,2,2.5,3,4,8]
codecs = ["h264", "hevc", "vp9"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_"
}


def visualize_results_by_codec(output_dir="visualizations"):
    
    for dataset in datasets:
        plt.figure(figsize=(12, 12))
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
                    "vmaf": [],
                    "tpsnr": [],
                    "tssim": [],
                    "fvd": []}
                for video_name, video_data in video_results.items():
                    metrics_for_codec[level]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    metrics_for_codec[level]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    metrics_for_codec[level]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    metrics_for_codec[level]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    metrics_for_codec[level]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    metrics_for_codec[level]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
            # average over all videos for each level
            bpp = []
            psn = []
            ssim = []
            vmaf = []
            tpsnr = []
            tssim = []
            fvd = []
            for level in levels:
                avg_psnr = np.mean(metrics_for_codec[level]["psnr"])
                avg_ssim = np.mean(metrics_for_codec[level]["ssim"])
                avg_vmaf = np.mean(metrics_for_codec[level]["vmaf"])
                avg_tpsnr = np.mean(metrics_for_codec[level]["tpsnr"])
                avg_tssim = np.mean(metrics_for_codec[level]["tssim"])
                avg_fvd = np.mean(metrics_for_codec[level]["fvd"])
                bpp_value = 0.1 * level  # assuming bpp increases linearly with level for simplicity
                bpp.append(bpp_value)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)

            plt.subplot(3, 2, 1)
            plt.plot(bpp, psn, marker='o', label=codec.upper())
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 2)
            plt.plot(bpp, ssim, marker='o', label=codec.upper())
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 3)
            plt.plot(bpp, vmaf, marker='o', label=codec.upper())
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 4)
            plt.plot(bpp, tpsnr, marker='o', label=codec.upper())
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 5)
            plt.plot(bpp, tssim, marker='o', label=codec.upper())
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 6)
            plt.plot(bpp, fvd, marker='o', label=codec.upper())
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
        
        plt.xlabel("Bits per Pixel (bpp)")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_codec.png"))
        plt.close()

def visualize_results_by_level(output_dir="visualizations"):
    for dataset in datasets:
        plt.figure(figsize=(12, 12))
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
                    "vmaf": [],
                    "tpsnr": [],
                    "tssim": [],
                    "fvd": []
                    }
                for video_name, video_data in video_results.items():
                    metrics_for_level[codec]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    metrics_for_level[codec]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    metrics_for_level[codec]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    metrics_for_level[codec]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    metrics_for_level[codec]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    metrics_for_level[codec]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
            # average over all videos for each codec
            codec_list = []
            psn = []
            ssim = []
            vmaf = []
            tpsnr = []
            tssim = []
            fvd = []
            for codec in codecs:
                avg_psnr = np.mean(metrics_for_level[codec]["psnr"])
                avg_ssim = np.mean(metrics_for_level[codec]["ssim"])
                avg_vmaf = np.mean(metrics_for_level[codec]["vmaf"])
                avg_tpsnr = np.mean(metrics_for_level[codec]["tpsnr"])
                avg_tssim = np.mean(metrics_for_level[codec]["tssim"])
                avg_fvd = np.mean(metrics_for_level[codec]["fvd"])
                codec_lower = codec.lower()
                codec_list.append(codec_lower)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)

            plt.subplot(3, 2, 1)
            plt.plot(codec_list, psn, marker='o', label=f"Level {level}")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 2)
            plt.plot(codec_list, ssim, marker='o', label=f"Level {level}")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 3)
            plt.plot(codec_list, vmaf, marker='o', label=f"Level {level}")
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 4)
            plt.plot(codec_list, tpsnr, marker='o', label=f"Level {level}")
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 5)
            plt.plot(codec_list, tssim, marker='o', label=f"Level {level}")
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(3, 2, 6)
            plt.plot(codec_list, fvd, marker='o', label=f"Level {level}")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()

        
        plt.xlabel("Codecs")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_level.png"))
        plt.close()