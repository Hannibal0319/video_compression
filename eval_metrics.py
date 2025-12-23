from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os

import argparse
import warnings
import json
import fvd_metric.fvd as fvd
from metrics_utils import compute_tSSIM_and_tPSNR_by_paths, tPSNR_by_paths, tSSIM_by_paths, \
                            compute_movie_index_by_paths, \
                            ST_RRED_by_paths \


from metrics.TI import TI_by_path



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
    base_name = "_".join(video_name.split("\\")[-1].split("_")[:-1]) + ".mp4"
    return base_name

datasets = ["BVI-HD"]
codecs = ["h264","hevc","vp9","av1"]
levels = ["1","1.5","2","2.5","3","4","8"]

compute_metrics =["psnr","ssim","vmaf","fvd","tssim","tpsnr","movie_index","st_rred"]

force = False
# is force is True we recompute all metrics even if they already exist

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    for codec in codecs:
        for level in levels:
            
            compressed_videos_path = os.path.join("compressed_videos", dataset, codec, level)
            if not os.path.exists(compressed_videos_path):
                print(f"Path not found, skipping: {compressed_videos_path}")
                continue

            compressed_videos = [video for video in os.listdir(compressed_videos_path) if not video.endswith(".json")]
            print(f"Found {len(compressed_videos)} compressed videos for {codec} at level {level}")
            
            json_output = {}

            for video in compressed_videos:
                compressed_video = os.path.join(compressed_videos_path, video)
                input_video = "videos/" + dataset + "/" + find_original_for_compressed(compressed_video)
                print("Processing", video, "with codec", codec, "at level", level)
                print("\nInput video:\n", input_video)
                print("\nCompressed video:\n", compressed_video)

                # skip if already computed
                json_path = f"results/eval_metrics_{dataset}_{codec}_level{level}.json"
                existing_data = {}
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {json_path}. Will create a new file.")
                
                if video in existing_data and all(metric in existing_data[video] for metric in compute_metrics) and not force:
                    print(f"Metrics for {video} already computed, skipping.")
                    continue
                
                if "psnr" in compute_metrics or "ssim" in compute_metrics or "vmaf" in compute_metrics and (not all(metric in existing_data.get(video, {}) for metric in compute_metrics) or force): 
                    ffqm = FfmpegQualityMetrics(input_video, compressed_video,verbose=True,progress=True,threads=os.cpu_count()-4 or 4)
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

                if "fvd" in compute_metrics and ("fvd" not in existing_data.get(video, {}) or force):
                    print("FVD")
                    fvd_value = fvd.fvd_pipeline(input_video, compressed_video)
                    print(fvd_value)
                    json_output[video]["fvd"] = fvd_value
                else:
                    print("Skipping FVD as already computed or not required.")

                if "tssim" in compute_metrics and "tpsnr" in compute_metrics and (not ("tssim" in existing_data.get(video, {}) and "tpsnr" in existing_data.get(video, {})) or force):
                    print("tSSIM and tPSNR")
                    temporal_ssim, temporal_psnr = compute_tSSIM_and_tPSNR_by_paths(input_video, compressed_video)
                    print("tSSIM:", temporal_ssim)
                    print("tPSNR:", temporal_psnr)
                    json_output[video]["tssim"] = temporal_ssim
                    json_output[video]["tpsnr"] = temporal_psnr
                elif "tssim" in compute_metrics and (not "tssim" in existing_data.get(video, {}) or force):
                    print("tSSIM")
                    temporal_ssim = tSSIM_by_paths(input_video, compressed_video)
                    print(temporal_ssim)
                    json_output[video]["tssim"] = temporal_ssim
                elif "tpsnr" in compute_metrics and (not "tpsnr" in existing_data.get(video, {}) or force):
                    print("tPSNR")
                    temporal_psnr = tPSNR_by_paths(input_video, compressed_video)
                    print(temporal_psnr)
                    json_output[video]["tpsnr"] = temporal_psnr
                

                if "movie_index" in compute_metrics and (not "movie_index" in existing_data.get(video, {}) or force):
                    print("Movie Index")
                    movie_index_value = compute_movie_index_by_paths(input_video, compressed_video)
                    print(movie_index_value)
                    json_output[video]["movie_index"] = movie_index_value
                else:
                    print("Skipping Movie Index as already computed or not required.")

                if "st_rred" in compute_metrics and (not "st_rred" in existing_data.get(video, {}) or force):
                    print("ST-RRED")
                    st_rred_value = ST_RRED_by_paths(input_video, compressed_video)
                    print(st_rred_value)
                    json_output[video]["st_rred"] = st_rred_value
                else:
                    print("Skipping ST-RRED as already computed or not required.")
                
            
                print()

            # write to file, but also do not delete other metrics already present
            if json_output: # Only write if there's new data
                print(f"Updating file: {json_path}")
                for video_name, metrics_data in json_output.items():
                    if video_name in existing_data:
                        existing_data[video_name].update(metrics_data)
                    else:
                        existing_data[video_name] = metrics_data
                
                with open(json_path, "w") as f:
                    json.dump(existing_data, f, indent=4)

    if "TI" in compute_metrics:
        print("Computing TI for dataset", dataset)
        videos = os.listdir("videos/" + dataset + "/")
        ti_results = {}
        for video in videos:
            if video.endswith(".y4m") or video.endswith(".mp4"):
                input_video = "videos/" + dataset + "/" + video
                print("Computing TI for video", video)
                ti_value = TI_by_path(input_video)
                ti_results[video] = ti_value
                print(f"TI for {video}: {ti_value}")
        
        with open(f"results/eval_metrics_{dataset}_TI.json", "w") as f:
            json.dump(ti_results, f, indent=4)


    
    
    print(f"Finished processing dataset {dataset}.\n")