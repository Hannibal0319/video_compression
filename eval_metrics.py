from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os

import argparse
import warnings
import json
import fvd_metric.fvd as fvd
from metrics_utils import tPSNR_by_paths, tSSIM_by_paths

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--print_all", action="store_true", help="Print all metrics")

def find_original_for_compressed(video_name):
    base_name = "_".join(video_name.split("/")[-1].split("_")[:-1]) + ".y4m"
    return base_name

datasets = ["UVG","HEVC_CLASS_B"]
codecs = ["h264","hevc","vp9"]
levels = ["1","1.5","2","2.5","3","4","8"]

for dataset in datasets:
    if not os.path.exists("results/eval_metrics_" + dataset.lower()):
        os.makedirs("results/eval_metrics_" + dataset.lower())
    compressed_videos =[] 
    for codec in codecs:
        for level in levels:
            compressed_videos.extend([(video,codec,level) for video in os.listdir("compressed_videos/" + dataset + "/" + codec + "/" + level + "/")])
    compressed_videos = list(filter(lambda x: not x[0].endswith(".json"), compressed_videos))
    


    json_output = {}
    compute_metrics =["psnr","ssim","vmaf"]  #,"fvd","tssim","tpsnr"]

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
            if video in existing_data and all(metric in existing_data[video] for metric in compute_metrics):
                print(f"Metrics for {video} with codec {codec} at level {level} already computed, skipping.")
                continue

        ffqm = FfmpegQualityMetrics(input_video, compressed_video,verbose=True,progress=True,threads=10)
        print("-"*40)
        print("Calculating metrics for", compressed_video)
        print("-"*40)
        json_output[video] = {}
        metrics = ffqm.calculate(compute_metrics)
        print("Metrics:", metrics.keys())
        print("PSNR")
        psnr_avg = sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"])
        json_output[video]["psnr"] = psnr_avg
        print(psnr_avg)
        print("SSIM")
        
        ssim_avg = sum([frame["ssim_avg"] for frame in metrics["ssim"]]) / len(metrics["ssim"])
        json_output[video]["ssim"] = ssim_avg
        print(ssim_avg)
        
        print("VMAF")
        vmaf_avg = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        json_output[video]["vmaf"] = vmaf_avg
        print(vmaf_avg)
        '''
        print("FVD")
        fvd_value = fvd.fvd_pipeline(input_video, compressed_video)
        print(fvd_value)
        json_output[video]["fvd"] = fvd_value
        
        print("tSSIM")
        temporal_ssim = tSSIM_by_paths(input_video, compressed_video)
        print(temporal_ssim)
        json_output[video]["tssim"] = temporal_ssim

        print("tPSNR")
        temporal_psnr = tPSNR_by_paths(input_video, compressed_video)
        print(temporal_psnr)
        json_output[video]["tpsnr"] = temporal_psnr
        '''
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