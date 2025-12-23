#cleanup compressed videos from Letter to Letter

import os


for codec in os.listdir("compressed_videos/BVI-HD"):
    codec_path = os.path.join("compressed_videos/BVI-HD", codec)
    for level in os.listdir(codec_path):
        level_path = os.path.join(codec_path, level)
        for filename in os.listdir(level_path):
            if filename.startswith("H") or filename.startswith("K") or filename.startswith("L") or filename.startswith("M") or filename.startswith("N") or filename.startswith("O") or filename.startswith("P"):
                file_path = os.path.join(level_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")