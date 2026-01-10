# Video Compression Benchmark

End-to-end pipeline to transcode reference video datasets with multiple codecs, compute quality metrics, and summarize rate–distortion performance with BD-scores.

## What’s Inside
- FFmpeg-based transcoding for `h264`, `hevc`, `vp9`, and `av1` with configurable bitrate levels via [transcode_pipeline.py](transcode_pipeline.py).
- Quality evaluation (PSNR, SSIM, VMAF, FVD, temporal SSIM/PSNR, Movie Index, ST-RRED, TI) via [eval_metrics.py](eval_metrics.py) and supporting utilities in [metrics_utils.py](metrics_utils.py).
- BD-score computation for cross-codec rate–distortion comparisons via [compute_BD_scores.py](compute_BD_scores.py).
- Dataset helpers: YUV➜Y4M conversion in [yuv2y4m.py](yuv2y4m.py); additional smoothness/entropy analyses in [energy.py](energy.py).

## Prerequisites
- Python 3.9+ (tested with PyTorch/TensorFlow packages listed in [requirements.txt](requirements.txt)).
- FFmpeg/ffprobe available on `PATH` (required for transcoding and ffmpeg-quality-metrics).
- GPU is recommended for FVD and some PyTorch/TensorFlow computations, but CPU-only runs are possible (slower).

## Setup
```bash
# optional: create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Data Layout
Place source videos under `videos/` using the expected dataset names:
- `videos/UVG/*.y4m` (convert YUV with [yuv2y4m.py](yuv2y4m.py) if needed)
- `videos/HEVC_CLASS_B/*.y4m` (converted from YUV)
- `videos/BVI-HD/*.mp4`

Compressed outputs are written to `compressed_videos/<DATASET>/<CODEC>/<LEVEL>/` and per-run metadata logs (`.json`) sit next to each file. Metric outputs live in `results/`.

## Quickstart
1) Convert YUV sources to Y4M (UVG/HEVC only):
```bash
python yuv2y4m.py
```

2) Transcode the datasets (defaults: all codecs, level list defined inside the script – currently `[12]`):
```bash
python transcode_pipeline.py --input_dir videos --output_dir compressed_videos --codecs h264 hevc vp9 av1 --workers 8 --vmaf
```
Edit the `levels` list inside `main()` if you need multiple bitrate points (e.g., `[1, 1.5, 2, 2.5, 3, 4, 8, 12]`).

3) Compute quality metrics (writes `results/eval_metrics_{DATASET}_{CODEC}_level{LEVEL}.json`):
```bash
python eval_metrics.py
```
The script currently uses in-file defaults (`datasets = ["BVI-HD","HEVC_CLASS_B","UVG"]`, `codecs = ["av1","h264","hevc","vp9"]`, `levels = ["12"]`, and `compute_metrics = ["fvd","st_rred","movie_index","tssim","tpsnr"]`). Adjust these lists near the top of the file to change coverage. Add `"TI"` to `compute_metrics` to emit temporal information JSON per dataset.

4) Derive BD-scores from the collected metrics (writes to `results/bd_scores/`):
```bash
python compute_BD_scores.py
```
By default this computes BD-rate curves for each dataset using `h264`, `hevc`, `vp9`, and `av1` as references; override `reference_codec` in `process_bd_scores()` if you want a single baseline.

5) (Optional) Run smoothness/entropy analyses:
```bash
python energy.py
```
This script expects prior TI results in `results/eval_metrics_{DATASET}_TI.json` and produces bending energy, cosine similarity, SVD entropy, and PCA-based smoothness summaries under `results/`.

## Notes and Tips
- Ensure FFmpeg builds include `libsvtav1`, `libx265`, `libvpx-vp9`, and `libx264` for the provided codec profiles.
- For large batches, consider lowering `--workers` in `transcode_pipeline.py` to avoid disk contention; VMAF and FVD are compute-intensive.
- If metrics already exist, `eval_metrics.py` will skip recomputation unless you set `force = True` inside the script.

## Project Structure (key files)
```
transcode_pipeline.py     # Batched FFmpeg transcoding and metadata sidecars
eval_metrics.py           # Quality metrics runner (PSNR/SSIM/VMAF/FVD/tSSIM/tPSNR/Movie Index/ST-RRED/TI)
compute_BD_scores.py      # BD-rate aggregation from results/
metrics_utils.py          # Metric implementations and helpers
yuv2y4m.py                # Dataset YUV→Y4M conversion helper
energy.py                 # Optional smoothness/entropy analyses
compressed_videos/        # Outputs organized by dataset/codec/level
results/                  # Metric JSONs and BD-score outputs
```

## License
Add your chosen license here (e.g., MIT).

## Contact
Questions or ideas? Open an issue or reach out to the maintainers.