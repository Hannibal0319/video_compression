from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os

def compute_video_metrics(original_video_path, compressed_video_path):
    """
    Compute video quality metrics between the original and compressed video.

    Parameters:
    original_video_path (str): Path to the original video file.
    compressed_video_path (str): Path to the compressed video file.

    Returns:
    dict: A dictionary containing computed metrics such as PSNR, SSIM, VMAF, etc.
    """
    ffqm = FfmpegQualityMetrics(original_video_path, compressed_video_path,verbose=True,progress=True)
    metrics = ffqm.calculate(["psnr"])
    psnr_avg = sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"])
    return psnr_avg

if __name__ == "__main__":
    original_video = "videos/BVI-HD/PlasmaSlowRandom_1920x1080_60fps.mp4"
    compressed_video = "compressed_videos/BVI-HD/vp9/12/PlasmaSlowRandom_1920x1080_60fps_vp9.webm"
    psnr_value = compute_video_metrics(original_video, compressed_video)
    print(f"Average PSNR: {min(psnr_value,60)}")
