# YUV to MP4 transcoding script
import subprocess
import cv2
import numpy as np
import tempfile
from pathlib import Path

def yuv2y4m(input_yuv, width, height, fps=30):
    output_y4m = str(Path(tempfile.gettempdir()) / (Path(input_yuv).stem + ".y4m"))
    
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
    all_yuv_videos = [
        str(p) for p in Path("videos").rglob("*.yuv") if p.is_file()
    ]
    print(f"Found {len(all_yuv_videos)} YUV files to convert.")
    for input_yuv in all_yuv_videos:
        print(f"Converting {input_yuv} to Y4M...")
        # assuming 1920x1080 resolution and 120 fps for all videos
        yuv2y4m(input_yuv, 1920, 1080, fps=120)
        print(f"Converted {input_yuv} to Y4M.")
        
        
        
