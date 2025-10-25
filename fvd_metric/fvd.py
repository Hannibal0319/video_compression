# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub



def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution

  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def create_id3_embedding(video_batch):

    """
    Convert a batch of videos to I3D embeddings (NumPy array)
    video_batch: [B, T, H, W, C] float32 in range [-1, 1] or [0, 1]
    """
    local = True
    if local:
        i3d_model = hub.KerasLayer(
        "./models/i3d-kinetics-400-v1",
        trainable=False
        )
    else:
        i3d_model = hub.KerasLayer(
        "https://tfhub.dev/deepmind/i3d-kinetics-400/1",
        trainable=False
        )
    if len(video_batch.shape) != 5:
        raise ValueError("Expected shape [B, T, H, W, C]")
    
    # Convert to tensor
    video_batch = tf.convert_to_tensor(video_batch, dtype=tf.float32)
    
    # Resize frames to 224x224
    B, T, H, W, C = video_batch.shape
    video_batch = tf.reshape(video_batch, (-1, H, W, C))  # merge batch and time
    video_batch = tf.image.resize(video_batch, (224, 224))
    video_batch = tf.reshape(video_batch, (B, T, 224, 224, C))
    video_batch_np = video_batch.numpy()  # works if in TF2 eager



    # Forward pass through KerasLayer (eager execution)
    embeddings = i3d_model(video_batch)
    embeddings_np = embeddings.numpy()  # convert to NumPy array
    return embeddings_np  # safe in eager mode

import numpy as np
from scipy import linalg

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid

def calculate_fvd(real_activations,
                  generated_activations):
    """Returns a list of ops that compute metrics as funcs of activations.

    Args:
    real_activations: <float32>[num_samples, embedding_size]
    generated_activations: <float32>[num_samples, embedding_size]

    Returns:
    A scalar that contains the requested FVD.
    """
    # Ensure numpy arrays and 2D shapes [N, D]
    real_activations = np.asarray(real_activations)
    generated_activations = np.asarray(generated_activations)
    if real_activations.ndim == 1:
        real_activations = real_activations[np.newaxis, :]
    if generated_activations.ndim == 1:
        generated_activations = generated_activations[np.newaxis, :]

    # Use float64 for numerical stability
    real_activations = real_activations.astype(np.float64)
    generated_activations = generated_activations.astype(np.float64)

    # Means
    mu1 = np.mean(real_activations, axis=0)
    mu2 = np.mean(generated_activations, axis=0)

    # Covariances: handle single-sample case explicitly (cov = zero matrix).
    if real_activations.shape[0] == 1:
        sigma1 = np.zeros((real_activations.shape[1], real_activations.shape[1]), dtype=np.float64)
    else:
        sigma1 = np.cov(real_activations, rowvar=False, ddof=0).astype(np.float64)

    if generated_activations.shape[0] == 1:
        sigma2 = np.zeros((generated_activations.shape[1], generated_activations.shape[1]), dtype=np.float64)
    else:
        sigma2 = np.cov(generated_activations, rowvar=False, ddof=0).astype(np.float64)

    # Sanitize any NaNs / Infs (defensive)
    sigma1 = np.nan_to_num(sigma1, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2 = np.nan_to_num(sigma2, nan=0.0, posinf=0.0, neginf=0.0)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)

    return float(fid)

def load_video(path):
    """Loads a video from a file.

    Args:
        path: The path to the video file.
    Returns:
        A 5D numpy array of shape [num_videos, num_frames, height, width, channels].
    """
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video = np.array(frames)
    video = video[np.newaxis, ...]  # Add batch dimension
    return video

def fvd_pipeline(real_videos_path, generated_videos_path):
    """Computes FVD between two single videos (one each)."""
    real_videos = load_video(real_videos_path)
    generated_videos = load_video(generated_videos_path)
    real_videos = tf.convert_to_tensor(real_videos, dtype=tf.float32)
    generated_videos = tf.convert_to_tensor(generated_videos, dtype=tf.float32)
    
    # Expect exactly one video per input (batch size 1) for pairwise comparison.
    if real_videos.shape[0] != 1 or generated_videos.shape[0] != 1:
      raise ValueError("fvd_pipeline expects exactly one video per input (batch size 1).")

    result = calculate_fvd(
        create_id3_embedding(preprocess(real_videos, (224, 224))),
        create_id3_embedding(preprocess(generated_videos, (224, 224)))
    )

    return result