
import os

datasets = ["BVI-HD"]

for dataset in datasets:
    for codec in os.listdir(f"compressed_videos/{dataset}"):
        codec_path = os.path.join(f"compressed_videos/{dataset}", codec)
        for level in os.listdir(codec_path):
            level_path = os.path.join(codec_path, level)
            for filename in os.listdir(level_path):
                # Check if the file size is 0 bytes
                file_path = os.path.join(level_path, filename)
                if os.path.getsize(file_path) == 0:
                    print(f"Removing empty file: {file_path}")
                    os.remove(file_path)