from matplotlib import pyplot as plt
import numpy as np
import os
import json
from TI_groups import get_TI_groups

datasets = ["UVG", "HEVC_CLASS_B"]
levels = [1,1.5,2,2.5,3,4,8]
codecs = ["h264", "hevc", "vp9","av1"]

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
                    "fvd": [],
                    "movie_index": [],
                    "st_rred": []}
                for video_name, video_data in video_results.items():
                    metrics_for_codec[level]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    metrics_for_codec[level]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    metrics_for_codec[level]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    metrics_for_codec[level]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    metrics_for_codec[level]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    metrics_for_codec[level]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
                    metrics_for_codec[level]["movie_index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                    metrics_for_codec[level]["st_rred"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
            # average over all videos for each level
            bpp = []
            psn = []
            ssim = []
            vmaf = []
            tpsnr = []
            tssim = []
            fvd = []
            movie_index = []
            st_rred = []
            for level in levels:
                avg_psnr = np.mean(metrics_for_codec[level]["psnr"])
                avg_ssim = np.mean(metrics_for_codec[level]["ssim"])
                avg_vmaf = np.mean(metrics_for_codec[level]["vmaf"])
                avg_tpsnr = np.mean(metrics_for_codec[level]["tpsnr"])
                avg_tssim = np.mean(metrics_for_codec[level]["tssim"])
                avg_fvd = np.mean(metrics_for_codec[level]["fvd"])
                avg_movie_index = np.mean(metrics_for_codec[level]["movie_index"])
                avg_st_rred = np.mean(metrics_for_codec[level]["st_rred"])
                bpp_value = 1000 * level  # assuming bpp increases linearly with level for simplicity
                bpp.append(bpp_value)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)
                movie_index.append(avg_movie_index)
                st_rred.append(avg_st_rred)

            plt.subplot(4, 2, 1)
            plt.plot(bpp, psn, marker='o', label=codec.upper())
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 2)
            plt.plot(bpp, ssim, marker='o', label=codec.upper())
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 3)
            plt.plot(bpp, vmaf, marker='o', label=codec.upper())
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 4)
            plt.plot(bpp, tpsnr, marker='o', label=codec.upper())
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 5)
            plt.plot(bpp, tssim, marker='o', label=codec.upper())
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 6)
            plt.plot(bpp, fvd, marker='o', label=codec.upper())
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 7)
            plt.plot(bpp, movie_index, marker='o', label=codec.upper())
            plt.ylabel("Movie Index")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 8)
            plt.plot(bpp, st_rred, marker='o', label=codec.upper())
            plt.ylabel("ST-RRED")
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
                    "fvd": [],
                    "movie_index": [],
                    "st_rred": []
                    }
                for video_name, video_data in video_results.items():
                    metrics_for_level[codec]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    metrics_for_level[codec]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    metrics_for_level[codec]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    metrics_for_level[codec]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    metrics_for_level[codec]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    metrics_for_level[codec]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
                    metrics_for_level[codec]["movie_index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                    metrics_for_level[codec]["st_rred"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
            
            # average over all videos for each codec
            codec_list = []
            psn = []
            ssim = []
            vmaf = []
            tpsnr = []
            tssim = []
            fvd = []
            movie_index = []
            st_rred = []
            for codec in codecs:
                avg_psnr = np.mean(metrics_for_level[codec]["psnr"])
                avg_ssim = np.mean(metrics_for_level[codec]["ssim"])
                avg_vmaf = np.mean(metrics_for_level[codec]["vmaf"])
                avg_tpsnr = np.mean(metrics_for_level[codec]["tpsnr"])
                avg_tssim = np.mean(metrics_for_level[codec]["tssim"])
                avg_fvd = np.mean(metrics_for_level[codec]["fvd"])
                avg_movie_index = np.mean(metrics_for_level[codec]["movie_index"])
                avg_st_rred = np.mean(metrics_for_level[codec]["st_rred"])
                codec_lower = codec.lower()
                codec_list.append(codec_lower)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)
                movie_index.append(avg_movie_index)
                st_rred.append(avg_st_rred)

            plt.subplot(4, 2, 1)
            plt.plot(codec_list, psn, marker='o', label=f"Level {level}")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 2)
            plt.plot(codec_list, ssim, marker='o', label=f"Level {level}")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 3)
            plt.plot(codec_list, vmaf, marker='o', label=f"Level {level}")
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 4)
            plt.plot(codec_list, tpsnr, marker='o', label=f"Level {level}")
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 5)
            plt.plot(codec_list, tssim, marker='o', label=f"Level {level}")
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 6)
            plt.plot(codec_list, fvd, marker='o', label=f"Level {level}")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 7)
            plt.plot(codec_list, movie_index, marker='o', label=f"Level {level}")
            plt.ylabel("Movie Index")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 8)
            plt.plot(codec_list, st_rred, marker='o', label=f"Level {level}")
            plt.ylabel("ST-RRED")
            plt.grid(True)
            plt.legend()
        
        plt.xlabel("Codecs")
        plt.savefig(os.path.join(output_dir, f"{dataset}_rd_curve_by_level.png"))
        plt.close()

def visalize_results_by_TI_group(output_dir="visualizations"):
    TI_groups = get_TI_groups(datasets)
    
    
            
    # Accumulate metrics for all videos in this codec across TI groups, datasets and levels
    for codec in codecs:
        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Rate-Distortion Curve by TI Groups for codec: {codec.upper()}")
        metrics_for_group = {}
        for group_id in TI_groups.keys():
            metrics_for_group[group_id] = {}
            for level in levels:
                metrics_for_group[group_id][level] = {
                    "psnr": [],
                    "ssim": [],
                    "vmaf": [],
                    "tpsnr": [],
                    "tssim": [],
                    "fvd": [],
                    "movie_index": [],
                    "st_rred": []
                }
        for dataset in datasets:
            for level in levels:                
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)
                for video_name, video_data in video_results.items():
                    for group_id, videos in TI_groups.items():

                        videos_by_first_word = [v[0].split("_")[0] for v in videos]
                        if video_name.split("_")[0] in videos_by_first_word:
                            metrics_for_group[group_id][level]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                            metrics_for_group[group_id][level]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                            metrics_for_group[group_id][level]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                            metrics_for_group[group_id][level]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                            metrics_for_group[group_id][level]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                            metrics_for_group[group_id][level]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
                            metrics_for_group[group_id][level]["movie_index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                            metrics_for_group[group_id][level]["st_rred"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
                            metrics_for_group[group_id][level]["bpp"] = 1000 * level  # assuming bpp increases linearly with level for simplicity
        groups_ids = list(TI_groups.keys())
        bpp = [metrics_for_group[groups_ids[0]][level]["bpp"] for level in levels]

        psn = []
        ssim = []
        vmaf = []
        tpsnr = []
        tssim = []
        fvd = []
        movie_index = []
        st_rred = []
        for group_id in groups_ids:
            psn = []
            ssim = []
            vmaf = []
            tpsnr = []
            tssim = []
            fvd = []
            movie_index = []
            st_rred = []
            for level in levels:
                avg_psnr = np.mean(metrics_for_group[group_id][level]["psnr"]) if len(metrics_for_group[group_id][level]["psnr"]) > 0 else 0
                avg_ssim = np.mean(metrics_for_group[group_id][level]["ssim"]) if len(metrics_for_group[group_id][level]["ssim"]) > 0 else 0
                avg_vmaf = np.mean(metrics_for_group[group_id][level]["vmaf"]) if len(metrics_for_group[group_id][level]["vmaf"]) > 0 else 0
                avg_tpsnr = np.mean(metrics_for_group[group_id][level]["tpsnr"]) if len(metrics_for_group[group_id][level]["tpsnr"]) > 0 else 0
                avg_tssim = np.mean(metrics_for_group[group_id][level]["tssim"]) if len(metrics_for_group[group_id][level]["tssim"]) > 0 else 0
                avg_fvd = np.mean(metrics_for_group[group_id][level]["fvd"]) if len(metrics_for_group[group_id][level]["fvd"]) > 0 else 0
                avg_movie_index = np.mean(metrics_for_group[group_id][level]["movie_index"]) if len(metrics_for_group[group_id][level]["movie_index"]) > 0 else 0
                avg_st_rred = np.mean(metrics_for_group[group_id][level]["st_rred"]) if len(metrics_for_group[group_id][level]["st_rred"]) > 0 else 0
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)
                movie_index.append(avg_movie_index)
                st_rred.append(avg_st_rred)
            plt.subplot(4, 2, 1)
            plt.plot(bpp, psn, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 2)
            plt.plot(bpp, ssim, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 3)
            plt.plot(bpp, vmaf, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 4)
            plt.plot(bpp, tpsnr, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 5)
            plt.plot(bpp, tssim, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 6)
            plt.plot(bpp, fvd, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 7)
            plt.plot(bpp, movie_index, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("Movie Index")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 8)
            plt.plot(bpp, st_rred, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("ST-RRED")
            plt.grid(True)
            plt.legend()

        plt.xlabel("TI Groups")
        plt.savefig(os.path.join(output_dir, f"rd_curve_by_TI_group_for_{codec}.png"))
        

if __name__ == "__main__":
    visualize_results_by_codec()
    visualize_results_by_level()
    visalize_results_by_TI_group()
    pass