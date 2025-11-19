#!/bin/bash

# add volume mount for the parent directory of mvs
docker run -it --ipc=host --env="DISPLAY" -v "C:\Users\Peter\Desktop\stuff\MIUN\quantitative_research\video_compression\:/app" -v /tmp/.X11-unix:/tmp/.X11-unix:rw --rm --name mv-extractor \
    mv-info