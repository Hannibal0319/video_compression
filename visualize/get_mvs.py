import av
import numpy as np

"""
should rebuild ffmpeg and python-av with motion vector support:
./configure --enable-libx264 --enable-debug=3 --extra-cflags=-g --extra-ldflags=-g
make -j8
pip install av --force-reinstall --no-binary av
"""


# Open video
container = av.open("compressed_videos/UVG/h264/3/Jockey_1920x1080_120fps_420_8bit_YUV_h264.mp4")

video_frames = []
c = 0
for frame in container.decode(video=0):
    if hasattr(frame, "motion_vectors") and frame.motion_vectors:
        for mv in frame.motion_vectors:
            video_frames.append({
                "source": mv.source,       # 0 = forward, 1 = backward
                "w": mv.w, "h": mv.h,      # block size
                "src_x": mv.src_x, "src_y": mv.src_y,
                "dst_x": mv.dst_x, "dst_y": mv.dst_y
            })
    else:
        print("No motion vectors found in frame",c)
    c += 1

# Convert to structured numpy array for easier processing
dtype = np.dtype([("source", np.int32), ("w", np.int32), ("h", np.int32),
                  ("src_x", np.int32), ("src_y", np.int32),
                  ("dst_x", np.int32), ("dst_y", np.int32)])



import matplotlib.pyplot as plt
import numpy as np
import cv2

#frames overlayed with motion vectors
frames=[]

# show motion vectors on video frames from index 0 to 10
container = av.open("compressed_videos\\UVG\\h264\\3\\Jockey_1920x1080_120fps_420_8bit_YUV_h264.mp4")
for i, frame in enumerate(container.decode(video=0)):
    if i > 10:
        break
    frame_img = np.array(frame.to_image())
    for mv in frame.motion_vectors:
        start = (mv.dst_x, mv.dst_y)
        end = (mv.src_x, mv.src_y)
        frame_img = cv2.arrowedLine(frame_img, start, end, color=(0, 255, 0), thickness=1, tipLength=0.3)
    
    frames.append(frame_img)
    
plt.figure(figsize=(10, 6))
for i, frame_img in enumerate(frames):
    plt.subplot(2, 5, i + 1)
    plt.imshow(frame_img)
    plt.axis('off')
    
plt.tight_layout()
plt.show()
    