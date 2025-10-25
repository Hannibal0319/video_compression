from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os

import argparse
import warnings
import json
import fvd_metric.fvd as fvd
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--print_all", action="store_true", help="Print all metrics")

# create an instance of FfmpegQualityMetrics
input_video = "videos/UVG/Jockey_1920x1080_120fps_420_8bit_YUV.y4m"

codec = "h264"
level = "3"  # 1, 2, 3, 4, 5

compressed_videos = [f for f in os.listdir("compressed_videos/UVG/" + codec + "/" + level + "/")]
print("\nCompressed videos:", compressed_videos, "\n")

json_output = {}

for e in compressed_videos:
    if "mkv" in e or "json" in e:
        continue
    compressed_video = "compressed_videos/UVG/" + codec + "/" + level + "/" + e
    ffqm = FfmpegQualityMetrics(input_video, compressed_video,verbose=True,progress=True,threads=10)
    print("-"*40)
    print("Calculating metrics for", compressed_video)
    print("-"*40)
    json_output[e] = {}
    '''
    metrics = ffqm.calculate(["psnr", "ssim", "vmaf"])
    print("Metrics:", metrics.keys())
    print("PSNR")
    psnr_avg = sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"])
    json_output[e]["psnr"] = psnr_avg
    print(psnr_avg)
    print("SSIM")
    ssim_avg = sum([frame["ssim_avg"] for frame in metrics["ssim"]]) / len(metrics["ssim"])
    json_output[e]["ssim"] = ssim_avg
    print(ssim_avg)
    print("VMAF")
    vmaf_avg = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
    json_output[e]["vmaf"] = vmaf_avg
    print(vmaf_avg)
    '''
    print("FVD")
    
    fvd_value = fvd.fvd_pipeline(input_video, compressed_video)
    
    print(fvd_value)
    json_output[e]["fvd"] = fvd_value
    print()

    with open("results/eval_metrics_uvg_" + codec + "_level" + level + ".json", "w") as f:
        json.dump(json_output, f, indent=4)