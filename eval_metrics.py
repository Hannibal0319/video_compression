from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os
# create an instance of FfmpegQualityMetrics
input_video = "videos\Jockey_1920x1080_120fps_420_8bit_YUV.mp4"

compressed_videos = [e for e in os.listdir("compressed_videos") if e.startswith("Jockey_1920x1080_120fps_420_8bit_YUV")]
print(compressed_videos)
for e in compressed_videos:
    if "mkv" in e or "json" in e:
        continue
    compressed_video = "compressed_videos/" + e
    ffqm = FfmpegQualityMetrics(input_video, compressed_video)

    metrics = ffqm.calculate(["ssim", "psnr", "vmaf"])
    print("Results for", compressed_video)

    print("SSIM")
    print(sum([frame["psnr_y"] for frame in metrics["psnr"]]) / len(metrics["psnr"]))    
    print("PSNR")
    print(sum([frame["ssim_y"] for frame in metrics["ssim"]]) / len(metrics["ssim"]))
    print("VMAF")
    print(sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"]))
    print()
