from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os
import torch
from fvd_metric import fvd_pipeline
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--print_all", action="store_true", help="Print all metrics")

# create an instance of FfmpegQualityMetrics
input_video = "videos/UVG/Jockey_1920x1080_120fps_420_8bit_YUV.y4m"

compressed_videos = [e for e in os.listdir("compressed_videos") if e.startswith("Jockey_1920x1080_120fps_420_8bit_YUV")]
print(compressed_videos)
for e in compressed_videos:
    if "mkv" in e or "json" in e:
        continue
    compressed_video = "compressed_videos/" + e
    ffqm = FfmpegQualityMetrics(input_video, compressed_video)
    print("Calculating metrics for", compressed_video)
    metrics = ffqm.calculate(["psnr"])

    print("PSNR")
    psnr_avg = sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"])
    print(psnr_avg)
    '''
    print("SSIM")
    ssim_avg = sum([frame["ssim_avg"] for frame in metrics["ssim"]]) / len(metrics["ssim"])
    print(ssim_avg)
    print("VMAF")
    vmaf_avg = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
    print(vmaf_avg)
    print("VIF")
    vif_avg = sum([frame["scale_0"] for frame in metrics["vif"]]) / len(metrics["vif"])
    print(vif_avg)
    
    print("FVD")
    fvd = fvd_pipeline(input_video, compressed_video)
    print(fvd)
    print()
    '''