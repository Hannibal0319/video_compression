from matplotlib import pyplot as plt
import numpy as np
import os
import json
from TI_groups import get_TI_groups

datasets = [ "BVI-HD", "HEVC_CLASS_B", "UVG"]
levels = [1,1.5,2,2.5,3,4,8,12]
codecs = ["h264", "hevc", "vp9", "av1"]

dataset_2_files = {
    "UVG": "results/eval_metrics_uvg_",
    "HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_",
    "BVI-HD": "results/eval_metrics_BVI-HD_"
}

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

                # average psnr, ssim, vmaf over videos plotted as line with respect to kbps 
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
            kbps = []
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
                kbps_value = 1000 * level  # assuming kbps increases linearly with level for simplicity
                kbps.append(kbps_value)
                psn.append(avg_psnr)
                ssim.append(avg_ssim)
                vmaf.append(avg_vmaf)
                tpsnr.append(avg_tpsnr)
                tssim.append(avg_tssim)
                fvd.append(avg_fvd)
                movie_index.append(avg_movie_index)
                st_rred.append(avg_st_rred)

            plt.subplot(4, 2, 1)
            plt.plot(kbps, psn, marker='o', label=codec.upper())
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 2)
            plt.plot(kbps, ssim, marker='o', label=codec.upper())
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 3)
            plt.plot(kbps, vmaf, marker='o', label=codec.upper())
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 4)
            plt.plot(kbps, tpsnr, marker='o', label=codec.upper())
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 5)
            plt.plot(kbps, tssim, marker='o', label=codec.upper())
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 6)
            plt.plot(kbps, fvd, marker='o', label=codec.upper())
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 7)
            plt.plot(kbps, movie_index, marker='o', label=codec.upper())
            plt.ylabel("Movie Index")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 8)
            plt.plot(kbps, st_rred, marker='o', label=codec.upper())
            plt.ylabel("ST-RRED")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Bits per Pixel (kbps)")
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

                # average psnr, ssim, vmaf over videos plotted as line with respect to kbps 
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

def visualize_results_by_TI_group(output_dir="visualizations"):
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
                            metrics_for_group[group_id][level]["kbps"] = 1000 * level  # assuming kbps increases linearly with level for simplicity
        groups_ids = list(TI_groups.keys())
        kbps = [metrics_for_group[groups_ids[0]][level]["kbps"] for level in levels]

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
            plt.plot(kbps, psn, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 2)
            plt.plot(kbps, ssim, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 3)
            plt.plot(kbps, vmaf, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("VMAF")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 4)
            plt.plot(kbps, tpsnr, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("tPSNR (dB)")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 5)
            plt.plot(kbps, tssim, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("tSSIM")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 6)
            plt.plot(kbps, fvd, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 7)
            plt.plot(kbps, movie_index, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("Movie Index")
            plt.grid(True)
            plt.legend()
            plt.subplot(4, 2, 8)
            plt.plot(kbps, st_rred, marker='o', label=f"TI Group {group_id}")
            plt.ylabel("ST-RRED")
            plt.grid(True)
            plt.legend()

        plt.xlabel("TI Groups")
        plt.savefig(os.path.join(output_dir, f"rd_curve_by_TI_group_for_{codec}.png"))

def visualize_results_by_TI_group_deviation_of_codecs(output_dir="visualizations", number_of_groups=4, fill_between=True):
    TI_groups = get_TI_groups(datasets, number_of_groups=number_of_groups)
    result_per_codec = {}
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
                            if (video_data["psnr"] if "psnr" in video_data else 0) > 100 or (video_data["psnr"] if "psnr" in video_data else 0) <= 0:
                                print(f"Warning: Unusual PSNR value {video_data['psnr']} for video {video_name}, codec {codec}, level {level}, dataset {dataset}")
                            metrics_for_group[group_id][level]["psnr"].append(min(video_data["psnr"] if "psnr" in video_data else 0, 60))
                            metrics_for_group[group_id][level]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                            metrics_for_group[group_id][level]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                            metrics_for_group[group_id][level]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                            metrics_for_group[group_id][level]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                            metrics_for_group[group_id][level]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
                            metrics_for_group[group_id][level]["movie_index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                            metrics_for_group[group_id][level]["st_rred"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
                            metrics_for_group[group_id][level]["kbps"] = 1000 * level  # assuming kbps increases linearly with level for simplicity
        groups_ids = list(TI_groups.keys())
        kbps = [metrics_for_group[groups_ids[0]][level]["kbps"] for level in levels]

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
        result_per_codec[codec] = {}
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

            result_per_codec[codec][group_id] = {
                "psnr": psn,
                "ssim": ssim,
                "vmaf": vmaf,
                "tpsnr": tpsnr,
                "tssim": tssim,
                "fvd": fvd,
                "movie_index": movie_index,
                "st_rred": st_rred
            }
        # Now plot deviation of codecs per TI group on a single figure

    # turn fvd to log scale for better visualization
    for codec in codecs:
        for group_id in TI_groups.keys():
            result_per_codec[codec][group_id]["fvd"] = [np.log10(x+1) if x>0 else 0 for x in result_per_codec[codec][group_id]["fvd"]]
    plt.figure(figsize=(6, 6))
    plt.suptitle("Rate-Distortion Curve showing min/max range across Codecs for each TI Group")
    kbps = [1000 * level for level in levels]
    metrics = ["psnr","ssim","vmaf", "fvd", "tpsnr", "tssim", "movie_index", "st_rred"]
    
    # Define colors for different TI groups to make the plot readable
    group_colors = plt.cm.get_cmap('tab10', len(TI_groups))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>','8']
    line_styles = ['--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-']
    for i, metric in enumerate(metrics):
        ax = plt.subplot(2, 4, i + 1)
        legend_handles = []
        
        for j, group_id in enumerate(TI_groups.keys()):
            color = group_colors(j)
            
            # Collect metric values for all codecs for the current group and metric
            all_codec_metrics = np.array([result_per_codec[codec][group_id][metric] for codec in codecs])
            
            # Find min and max across codecs for each kbps level
            min_values = np.min(all_codec_metrics, axis=0)
            max_values = np.max(all_codec_metrics, axis=0)
            # say if there is any nan value in min_values or max_values, print a warning
            if np.isnan(min_values).any() or np.isnan(max_values).any():
                print(f"Warning: NaN values found in min or max values for Group {group_id}, Metric {metric}")
                # search for which codec and level has nan value
                for codec in codecs:
                    codec_metrics = result_per_codec[codec][group_id][metric]
                    for level_idx, value in enumerate(codec_metrics):
                        if np.isnan(value):
                            print(f"NaN found for Codec {codec}, Group {group_id}, Metric {metric}, Level {levels[level_idx]}")

            # Plot the min and max lines for the current group
            ax.plot(kbps, min_values, marker=markers[j], linestyle=line_styles[j], color=color)
            ax.plot(kbps, max_values, marker=markers[j], linestyle=line_styles[j], color=color)
            
            # Fill the area between min and max for the current group
            if fill_between: ax.fill_between(kbps, min_values, max_values, alpha=0.4, color=color, label=f'TI Group {group_id} Range')

            # Create a proxy artist for the legend
            legend_handles.append(plt.Line2D([0], [0], marker=markers[j], color='w', label=f'TI Group {group_id} Range',
                               markerfacecolor=color, markersize=10))
        ax.set_ylabel(metric.upper())
        ax.grid(True)
        ax.legend(handles=legend_handles)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "rd_curve_range_all_TI_groups.png"))
    plt.close()

    # another plot for the average curve across codecs per TI group
    plt.figure(figsize=(12, 18))
    #plt.suptitle("Rate-Distortion Curve showing Average across Codecs for each TI Group")
    for i, metric in enumerate(metrics):
        ax = plt.subplot(4, 2, i + 1)
        legend_handles = []
        
        for j, group_id in enumerate(TI_groups.keys()):
            color = group_colors(j)
            
            # Collect metric values for all codecs for the current group and metric
            all_codec_metrics = np.array([result_per_codec[codec][group_id][metric] for codec in codecs])
            
            # Find average across codecs for each kbps level
            avg_values = np.mean(all_codec_metrics, axis=0)
            print(f"Group {group_id}, Metric {metric}, Avg Values: {avg_values}")

            # Plot the average line for the current group
            ax.plot(kbps, avg_values, marker=markers[j], linestyle=line_styles[j], color=color, label=f'TI Group {group_id} Average')

            # Create a proxy artist for the legend
            legend_handles.append(plt.Line2D([0], [0], marker=markers[j], color='w', label=f'TI Group {group_id} Average',
                               markerfacecolor=color, markersize=10))
        ax.set_ylabel(metric.upper())
        ax.grid(True)
        ax.legend(handles=legend_handles, loc=get_loc_by_metric(metric),frameon=True,framealpha=0.8,bbox_to_anchor=(0.0, 0.83) if metric=="movie_index" else None)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "rd_curve_average_all_TI_groups.png"))
    plt.close()

def get_loc_by_metric(metric_name):
    metric_locs = {
        "psnr": "lower right",
        "ssim": "lower right",
        "vmaf": "lower right",
        "tpsnr": "lower right",
        "tssim": "lower right",
        "fvd": "upper right",
        "movie_index": "upper left",
        "st_rred": "upper right"
    }
    return metric_locs.get(metric_name.lower(), None)

def visualize_results_by_video(output_dir="visualizations/plots_by_video"):
    for dataset in datasets:
        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Rate-Distortion Curve for {dataset}")

        metrics_for_video_by_level = {}
        for codec in codecs:
            for level in levels:
                results_file = f"{dataset_2_files[dataset]}{codec}_level{level}.json"
                with open(results_file, 'r') as f:
                    video_results = json.load(f)

                for video_name, video_data in video_results.items():
                    #print(video_name)
                    video_name = "_".join(video_name.split("_")[:-1])
                    if video_name not in metrics_for_video_by_level:
                        metrics_for_video_by_level[video_name] = {}
                    
                    if level not in metrics_for_video_by_level[video_name]:
                        metrics_for_video_by_level[video_name][level] = {}
                    
                    if codec not in metrics_for_video_by_level[video_name][level]:
                        metrics_for_video_by_level[video_name][level][codec] = {
                            "psnr": [],
                            "ssim": [],
                            "vmaf": [],
                            "tpsnr": [],
                            "tssim": [],
                            "fvd": [],
                            "movie_index": [],
                            "st_rred": []
                        }
                    metrics_for_video_by_level[video_name][level][codec]["psnr"].append(video_data["psnr"] if "psnr" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["ssim"].append(video_data["ssim"] if "ssim" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["vmaf"].append(video_data["vmaf"] if "vmaf" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["tpsnr"].append(video_data["tpsnr"] if "tpsnr" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["tssim"].append(video_data["tssim"] if "tssim" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["fvd"].append(video_data["fvd"] if "fvd" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["movie_index"].append(video_data["movie_index"] if "movie_index" in video_data else 0)
                    metrics_for_video_by_level[video_name][level][codec]["st_rred"].append(video_data["st_rred"] if "st_rred" in video_data else 0)
        #print(list(metrics_for_video_by_level.values())[0].values())
        for video_name, metrics_by_level_codec in metrics_for_video_by_level.items():
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Rate-Distortion Curve for {video_name} in {dataset}")
            
            print(f"Plotting RD curve for video: {video_name} in dataset: {dataset}")
            print(metrics_by_level_codec)
            for codec in codecs:
                kbps = []
                psn = []
                ssim = []
                vmaf = []
                tpsnr = []
                tssim = []
                fvd = []
                movie_index = []
                st_rred = []
                for level in levels:
                    kbps.append(1000 * level)  # assuming kbps increases linearly with level for simplicity
                    psn.append(metrics_by_level_codec[level][codec]["psnr"][0])
                    ssim.append(metrics_by_level_codec[level][codec]["ssim"][0])
                    vmaf.append(metrics_by_level_codec[level][codec]["vmaf"][0])
                    tpsnr.append(metrics_by_level_codec[level][codec]["tpsnr"][0])
                    tssim.append(metrics_by_level_codec[level][codec]["tssim"][0])
                    fvd.append(metrics_by_level_codec[level][codec]["fvd"][0])
                    movie_index.append(metrics_by_level_codec[level][codec]["movie_index"][0])
                    st_rred.append(metrics_by_level_codec[level][codec]["st_rred"][0])

                print(f"Codec: {codec}, kbps: {kbps}, PSNR: {psn}")
                plt.subplot(4, 2, 1)
                plt.plot(kbps, psn, marker='o',label=codec.upper())
                plt.ylabel("PSNR (dB)")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 2)
                plt.plot(kbps, ssim, marker='o',label=codec.upper())
                plt.ylabel("SSIM")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 3)
                plt.plot(kbps, vmaf, marker='o',label=codec.upper())
                plt.ylabel("VMAF")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 4)
                plt.plot(kbps, tpsnr, marker='o',label=codec.upper())
                plt.ylabel("tPSNR (dB)")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 5)
                plt.plot(kbps, tssim, marker='o',label=codec.upper())
                plt.ylabel("tSSIM")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 6)
                plt.plot(kbps, fvd, marker='o',label=codec.upper())
                plt.ylabel("FVD")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 7)
                plt.plot(kbps, movie_index, marker='o',label=codec.upper())
                plt.ylabel("Movie Index")
                plt.grid(True)
                plt.legend()
                plt.subplot(4, 2, 8)
                plt.plot(kbps, st_rred, marker='o',label=codec.upper())
                plt.ylabel("ST-RRED")
                plt.grid(True)
                plt.legend()
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            os.makedirs(f"{output_dir}/{dataset}", exist_ok=True)
            plt.savefig(f"{output_dir}/{dataset}/{video_name}_rd_curve.png")
            plt.close()
    

if __name__ == "__main__":
    #visualize_results_by_codec()
    #visualize_results_by_level()
    #visualize_results_by_TI_group()
    #visualize_results_by_video()
    visualize_results_by_TI_group_deviation_of_codecs(number_of_groups=4, fill_between=True)
    pass