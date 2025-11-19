from mvextractor.videocap import VideoCap

cap = VideoCap("./compressed_videos/UVG/h264/1/Bosphorus_1920x1080_120fps_420_8bit_YUV_h264.mp4")
cap.open(".mp4")


while True:
    ret, frame, motion_vectors, frame_type = cap.read()
    print ("----------------------------")
    print (f"Read frame: {ret}")
    if not ret:
        break
    print(f"Num. motion vectors: {len(motion_vectors)}")
    print(f"Frame type: {frame_type}")
    if frame is not None:
        print(f"Frame size: {frame.shape}")

cap.release()
