# YUV to MP4 transcoding script
import cv2
import numpy as np
import tempfile
from pathlib import Path

def yuv2mp4(input_yuv, output_mp4):
    width = 1920
    height = 1080
    yuv_frame_size = width * height * 3 // 2  # YUV 4:2:0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_mp4, fourcc, 30.0, (width, height))

    with open(input_yuv, 'rb') as f:
        while True:
            yuv = f.read(yuv_frame_size)
            if len(yuv) < yuv_frame_size:
                break
            y = np.frombuffer(yuv[0:width*height], dtype=np.uint8).reshape((height, width))
            u = np.frombuffer(yuv[width*height:width*height+(width//2)*(height//2)], dtype=np.uint8).reshape((height//2, width//2))
            v = np.frombuffer(yuv[width*height+(width//2)*(height//2):], dtype=np.uint8).reshape((height//2, width//2))
            # Upsample U and V
            u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
            yuv_img = cv2.merge((y, u_up, v_up))
            frame = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
            out.write(frame)
    out.release()
 
    
if __name__ == "__main__":
    # all yuv in videos directory
    all_yuv_videos = [
        str(p) for p in Path("videos").rglob("*.yuv") if p.is_file()
    ]
    print(f"Found {len(all_yuv_videos)} YUV files to convert.")
    for input_yuv in all_yuv_videos:
        output_mp4 = f"{Path(input_yuv).stem}.mp4"  # Path to output MP4 file
        yuv2mp4(input_yuv, output_mp4)