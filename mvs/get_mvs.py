import numpy as np
import cv2
import subprocess
import shlex
import re

def get_mvs(video_path):
    """Extract motion vectors from a video file using ffmpeg's export_mvs.
    Returns a list of (frame_number, np.ndarray of shape (N,4)) where each row
    is (x, y, dx, dy). This parser is heuristic and may need adjustment for
    different ffmpeg builds/formats. For robust access prefer PyAV and the
    frame side_data 'motion_vectors' if available.
    """
    cmd = [
        "ffmpeg",
        "-flags2", "+export_mvs",
        "-loglevel", "debug",
        "-i", str(video_path),
        "-vf", "codecview=mv=pf+bf+bb",
        "-f", "null",
        "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, stderr = process.communicate()

    frame_mvs = []
    current_frame = None
    current_mvs = []

    # Heuristic parsing: look for frame markers and groups of 4 numbers (x y dx dy)
    for line in stderr.splitlines():
        # detect frame number lines like "frame=  23" (common ffmpeg progress lines)
        mframe = re.search(r"\bframe=\s*(\d+)\b", line)
        if mframe:
            # if we were collecting MVs for previous frame, save them
            if current_frame is not None and current_mvs:
                frame_mvs.append((current_frame, np.array(current_mvs, dtype=float)))
                current_mvs = []
            current_frame = int(mframe.group(1))

        # find numeric groups of 4 (x y dx dy). Matches floats or ints, separated by spaces or commas
        quads = re.findall(r"(-?\d+\.?\d+)[, ]+(-?\d+\.?\d+)[, ]+(-?\d+\.?\d+)[, ]+(-?\d+\.?\d+)", line)
        # filter out accidental matches by requiring plausible motion-vector magnitude (optional)
        for q in quads:
            x, y, dx, dy = map(float, q)
            current_mvs.append((x, y, dx, dy))

    # append last collected frame mvs
    if current_frame is not None and current_mvs:
        frame_mvs.append((current_frame, np.array(current_mvs, dtype=float)))

    return frame_mvs

print(get_mvs("compressed_videos/UVG/h264/1/Jockey_1920x1080_120fps_420_8bit_YUV_h264.mp4"))