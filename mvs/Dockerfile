# Dockerfile to extract motion vectors using mvs
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    cmake \
    libopencv-dev

# Install required Python packages
RUN git clone https://github.com/jishnujayakumar/MV-Tractus.git
RUN cd MV-Tractus
RUN ./install_ffmpeg.sh
RUN sudo cp ffmpeg.conf /etc/ld.so.conf.d/ffmpeg.conf && sudo ldconfig
RUN ./compile

ENTRYPOINT [ "/bin/bash" ]