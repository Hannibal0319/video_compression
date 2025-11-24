import numpy as np
import concurrent.futures
import os
from metrics.load_video_frames import load_video_frames

def temporal_information(frames: np.ndarray) -> float:
    """
    Calculate the Temporal Information (TI) of a video sequence.

    Parameters:
    frames (np.ndarray): A 4D numpy array of shape (num_frames, height, width, channels)
                         representing the video frames.

    Returns:
    float: The Temporal Information value.
    """
    num_frames, height, width, channels = frames.shape
    ti_values = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4 or 4) as executor:
        futures = []
        for t in range(1, num_frames):
            futures.append(executor.submit(np.std, frames[t].astype(np.float32) - frames[t - 1].astype(np.float32)))
        for future in concurrent.futures.as_completed(futures):
            ti_values.append(future.result())

    # Return the average TI over all frame differences
    return float(np.mean(ti_values)) if ti_values else 0.0

def TI_by_path(input_video_path: str) -> float:
    """
    Compute the Temporal Information (TI) of an input video.

    Parameters:
    input_video_path (str): Path to the original input video file.

    Returns:
    float: The Temporal Information value of the input video.
    """

    frames = load_video_frames(input_video_path)
    ti_value = temporal_information(frames)
    return ti_value