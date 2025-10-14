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


import six
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
    
    
    mu1, sigma1 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid

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
    """Computes FVD between two sets of videos.
    """
    real_videos = load_video(real_videos_path)
    generated_videos = load_video(generated_videos_path)
    real_videos = tf.convert_to_tensor(real_videos, dtype=tf.float32)
    generated_videos = tf.convert_to_tensor(generated_videos, dtype=tf.float32)
    

    if real_videos.shape[0] != generated_videos.shape[0]:
      raise ValueError("The number of videos must be the same for both sets.")

    real_activations = []
    generated_activations = []

    real_preprocessed = preprocess(real_videos, (224, 224))
    generated_preprocessed = preprocess(generated_videos, (224, 224))
    real_embeddings = create_id3_embedding(real_preprocessed)
    generated_embeddings = create_id3_embedding(generated_preprocessed)
    real_activations.append(real_embeddings)
    generated_activations.append(generated_embeddings)

    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)

    return calculate_fvd(real_activations, generated_activations)