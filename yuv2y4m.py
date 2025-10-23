# YUV to MP4 transcoding script
import subprocess
import cv2
import numpy as np
from pathlib import Path

def yuv2y4m(input_yuv, width, height, fps=30):
    output_y4m = str(Path(input_yuv).with_suffix(".y4m"))

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-framerate", str(fps), "-video_size", f"{width}x{height}",
        "-pix_fmt", "yuv420p",
        "-i", str(input_yuv),
        output_y4m
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # all yuv in videos directory
    UVG_Videos = [
        str(p) for p in Path("videos/UVG").rglob("*.yuv") if p.is_file()
    ]
    print(f"Found {len(UVG_Videos)} YUV files to convert.")
    HEVC_Videos = [
        str(p) for p in Path("videos/HEVC_CLASS_B").rglob("*.yuv") if p.is_file()
    ]
    for input_yuv in UVG_Videos:
        print(f"Converting {input_yuv} to Y4M...")
        # assuming 1920x1080 resolution and 120 fps for all videos
        yuv2y4m(input_yuv, 1920, 1080, fps=120)
        print(f"Converted {input_yuv} to Y4M.")

    for input_yuv in HEVC_Videos:
        # assuming 1920x1080
        fps = input_yuv.split("_")[-1].replace(".yuv", "")
        print(f"Assuming fps={fps} for {input_yuv}")
        yuv2y4m(input_yuv, 1920, 1080, fps=fps)
        print(f"Converted {input_yuv} to Y4M.")
