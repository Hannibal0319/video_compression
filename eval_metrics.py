from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os

import argparse
import warnings
import json
import fvd_metric.fvd as fvd
from metrics_utils import \
                            \
                            compute_movie_index_by_paths, \
                            ST_RRED_by_paths

from metrics.TPSNR_and_TSSIM import compute_tSSIM_and_tPSNR_by_paths, tPSNR_by_paths, tSSIM_by_paths

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--print_all", action="store_true", help="Print all metrics")
arg_parser.add_argument("--datasets", type=str, nargs="+", default=["UVG","HEVC_CLASS_B"], help="Datasets to evaluate on")
arg_parser.add_argument("--codecs", type=str, nargs="+", default=["h264","hevc","vp9"], help="Codecs to evaluate")
arg_parser.add_argument("--levels", type=str, nargs="+", default=["1","1.5","2","2.5","3","4","8"], help="Compression levels to evaluate")
arg_parser.add_argument("--metrics", type=str, nargs="+", default=["psnr","ssim","vmaf","fvd","tssim","tpsnr"], help="Metrics to compute")

args = arg_parser.parse_args()

def find_original_for_compressed(video_name):
    base_name = "_".join(video_name.split("/")[-1].split("_")[:-1]) + ".y4m"
    return base_name

datasets = ["UVG"]
codecs = ["h264","hevc","vp9"]
levels = ["1","1.5","2","2.5"]

compute_metrics =["tpsnr","tssim"]

force = True
# is force is True we recompute all metrics even if they already exist

for dataset in datasets:
    if not os.path.exists("results/eval_metrics_" + dataset.lower()):
        os.makedirs("results/eval_metrics_" + dataset.lower())
    compressed_videos =[] 
    for codec in codecs:
        for level in levels:
            compressed_videos.extend([(video,codec,level) for video in os.listdir("compressed_videos/" + dataset + "/" + codec + "/" + level + "/")])
    compressed_videos = list(filter(lambda x: not x[0].endswith(".json"), compressed_videos))
    print(f"Found compressed videos for dataset {dataset}: {len(compressed_videos)}")
    json_output = {}

    for tuple in compressed_videos:
        video, codec, level = tuple
        
        compressed_video = "compressed_videos/"+ dataset +"/" + codec + "/" + level + "/" + video
        input_video = "videos/" + dataset + "/" + find_original_for_compressed(compressed_video)
        print("Processing", video, "with codec", codec, "at level", level)
        print("\nInput video:\n", input_video)
        print("\nCompressed video:\n", compressed_video)
        # skip if already computed
        if os.path.exists("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json"):
            with open("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json", "r") as f:
                existing_data = json.load(f)
            if video in existing_data and all(metric in existing_data[video] for metric in compute_metrics) and not force:
                print(f"Metrics for {video} with codec {codec} at level {level} already computed, skipping.")
                continue
        else:
            print(f"No existing metrics file for codec {codec} at level {level}, will create new.")
            existing_data = {}
        
        if "psnr" in compute_metrics or "ssim" in compute_metrics or "vmaf" in compute_metrics and (not all(metric in existing_data.get(video, {}) for metric in compute_metrics) or force): 
            ffqm = FfmpegQualityMetrics(input_video, compressed_video,verbose=True,progress=True,threads=10)
        print("-"*40)
        print("Calculating metrics for", compressed_video)
        
        print("-"*40)
        json_output[video] = {}
        ffqm_metrics = list(filter(lambda x: x in ["psnr","ssim","vmaf"], compute_metrics))

        if ("psnr" in ffqm_metrics or "ssim" in ffqm_metrics or "vmaf" in ffqm_metrics) and (not all(metric in existing_data.get(video, {}) for metric in ffqm_metrics) or force):
            print("Calculating ffqm metrics:", ffqm_metrics)
            metrics = ffqm.calculate(ffqm_metrics)
            print("Metrics:", metrics.keys())
        else:
            print("Skipping ffqm metrics calculation as all are already computed or not required.")
        
        if "psnr" in compute_metrics and ("psnr" not in existing_data.get(video, {}) or force):
            print("PSNR")
            psnr_avg = sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"])
            json_output[video]["psnr"] = psnr_avg
            print(psnr_avg)
        else:
            print("Skipping PSNR as already computed or not required.")

        if "ssim" in compute_metrics and ("ssim" not in existing_data.get(video, {}) or force):
            print("SSIM")
            ssim_avg = sum([frame["ssim_avg"] for frame in metrics["ssim"]]) / len(metrics["ssim"])
            json_output[video]["ssim"] = ssim_avg
            print(ssim_avg)
        else:
            print("Skipping SSIM as already computed or not required.")

        if "vmaf" in compute_metrics and ("vmaf" not in existing_data.get(video, {}) or force):
            print("VMAF")
            vmaf_avg = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
            json_output[video]["vmaf"] = vmaf_avg
            print(vmaf_avg)
        else:
            print("Skipping VMAF as already computed or not required.")

        if "fvd" in compute_metrics and "fvd" not in existing_data.get(video, {}):
            print("FVD")
            fvd_value = fvd.fvd_pipeline(input_video, compressed_video)
            print(fvd_value)
            json_output[video]["fvd"] = fvd_value
        else:
            print("Skipping FVD as already computed or not required.")

        if "tssim" in compute_metrics and "tpsnr" in compute_metrics and not ("tssim" in existing_data.get(video, {}) and "tpsnr" in existing_data.get(video, {})):
            print("tSSIM and tPSNR")
            temporal_ssim, temporal_psnr = compute_tSSIM_and_tPSNR_by_paths(input_video, compressed_video)
            print("tSSIM:", temporal_ssim)
            print("tPSNR:", temporal_psnr)
            json_output[video]["tssim"] = temporal_ssim
            json_output[video]["tpsnr"] = temporal_psnr
        elif "tssim" in compute_metrics and "tssim" not in existing_data.get(video, {}):
            print("tSSIM")
            temporal_ssim = tSSIM_by_paths(input_video, compressed_video)
            print(temporal_ssim)
            json_output[video]["tssim"] = temporal_ssim
        elif "tpsnr" in compute_metrics and "tpsnr" not in existing_data.get(video, {}):
            print("tPSNR")
            temporal_psnr = tPSNR_by_paths(input_video, compressed_video)
            print(temporal_psnr)
            json_output[video]["tpsnr"] = temporal_psnr
        

        if "movie_index" in compute_metrics and "movie_index" not in existing_data.get(video, {}):
            print("Movie Index")
            movie_index_value = compute_movie_index_by_paths(input_video, compressed_video)
            print(movie_index_value)
            json_output[video]["movie_index"] = movie_index_value
        else:
            print("Skipping Movie Index as already computed or not required.")

        if "st_rred" in compute_metrics and "st_rred" not in existing_data.get(video, {}):
            print("ST-RRED")
            st_rred_value = ST_RRED_by_paths(input_video, compressed_video)
            print(st_rred_value)
            json_output[video]["st_rred"] = st_rred_value
        else:
            print("Skipping ST-RRED as already computed or not required.")
        
       
        print()

        # write to file, but also do not delete other metrics already present
        if os.path.exists("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json"):
            print("File named results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json exists, updating it.")
            
            with open("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json", "r") as f:
                existing_data = json.load(f)
            
            with open("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json", "w") as f:
                for video_name in json_output:
                    if video_name in existing_data:
                        for key in json_output[video_name]:
                            existing_data[video_name][key] = json_output[video_name][key]
                    else:
                        existing_data[video_name] = json_output[video_name]
                json.dump(existing_data, f, indent=4)
        else:
            print("Creating new file results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json.")
            with open("results/eval_metrics_" + dataset + "_" + codec + "_level" + level + ".json", "w") as f:
                json.dump(json_output, f, indent=4)