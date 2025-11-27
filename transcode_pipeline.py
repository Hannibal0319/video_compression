import argparse, concurrent.futures as fut, json, os, shlex, subprocess, sys, time
from pathlib import Path


CODEC_PROFILES = {
    "h264": {
        "ext": "mp4",
        "args": '-c:v libx264 -preset slow -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k'
    },
    "hevc": {
        "ext": "mp4",
        "args": '-c:v libx265 -preset slow -pix_fmt yuv420p10le -x265-params "aq-mode=3" -movflags +faststart -c:a aac -b:a 128k'
    },
    "vp9": {
        "ext": "webm",
        "args": '-c:v libvpx-vp9 -row-mt 1 -c:a libopus -b:a 128k'
    },
    "av1": {
        "ext": "mkv",
        "args": '-c:v libsvtav1 -probesize 50M -analyzeduration 100M -preset 4 -svtav1-params rc=1 -c:a libopus -b:a 128k'
    },
}

def compression_level_params(level):
    bandwidth = level * 1000  # in kbps
    return f"-b:v {bandwidth}k -maxrate {bandwidth}k -bufsize {bandwidth}k"

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def ffprobe_json(path):
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(str(path))}'
    rc, out = run(cmd)
    return json.loads(out) if rc == 0 else {}

def transcode_one(src, dst_dir, codec, vmaf_ref=None,level=2):
    prof = CODEC_PROFILES[codec]
    out_path = (dst_dir / (src.stem + f'_{codec}.' + prof["ext"])).absolute()
    os.makedirs(dst_dir, exist_ok=True)
    src_abs = src.absolute()
    level_params = compression_level_params(level) if codec != "av1" else f"-b:v {level*1000}k"
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(src_abs),
        *prof["args"].split(),
        *level_params.split(),
        str(out_path)
    ]
    print(f"Transcoding {src_abs} to {out_path} using {codec}...")
    print(cmd)
    t0 = time.time()

    rc, log = run(cmd)
    meta = {
        "source": str(src_abs),
        "output": str(out_path),
        "codec": codec,
        "ok": log,
        "seconds": time.time() - t0,
    }
    if rc == 0:
        meta["probe_out"] = ffprobe_json(out_path)
    # write sidecar
    with open(out_path.with_suffix(out_path.suffix + '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta, log

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", help="Input directory containing YUV files",default="videos")
    ap.add_argument("--output_dir", help="Output directory for transcoded videos", default="compressed_videos")
    ap.add_argument("--codecs","-c", nargs="+", default=["h264" ,"hevc","vp9","av1"], choices=CODEC_PROFILES.keys())
    ap.add_argument("--workers", type=int, default=os.cpu_count()-4 or 4)
    ap.add_argument("--vmaf", action="store_true", help="Compute VMAF vs original")
    args = ap.parse_args()
    
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    
    in_dir_UVG = in_dir / "UVG"
    in_dir_HEVC = in_dir / "HEVC_CLASS_B"
    
    videos_UVG = [p for p in in_dir_UVG.rglob("*.y4m")]
    videos_HEVC = [p for p in in_dir_HEVC.rglob("*.y4m")]

    if not videos_UVG:
        print("No input videos found.", file=sys.stderr)
        sys.exit(2)
    if not videos_HEVC:
        print("No input videos found.", file=sys.stderr)
        sys.exit(2)

    tasks = []
    videos = [videos_UVG, videos_HEVC]
    in_dirs = [in_dir_UVG, in_dir_HEVC]
    datasets = ["UVG","HEVC_CLASS_B"]
    levels = [1,1.5, 2,2.5,3, 4, 8]

    for i in range(len(videos)):
        print(f"Transcoding {len(videos[i])} videos to {args.codecs} using {args.workers} workers...")
        with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for src in videos[i]:
                
                for level in levels:
                    for codec in args.codecs:
                        if os.path.exists(out_dir / datasets[i] / codec / str(level) / src.relative_to(in_dirs[i]).parent / (src.stem + f'_{codec}.' + CODEC_PROFILES[codec]["ext"])):
                            print(f"Skipping existing file for {src} at level {level} with codec {codec}")
                            continue
                        dst_sub = out_dir / datasets[i] / codec / str(level) / src.relative_to(in_dirs[i]).parent
                        tasks.append(ex.submit(transcode_one, src, dst_sub, codec, vmaf_ref=str(src) if args.vmaf else None, level=level))
            
            for t in fut.as_completed(tasks):
                meta, _ = t.result()
                print(json.dumps(meta, indent=2))

def main_only_one():
    src = Path("videos/UVG/Beauty_1920x1080_120fps_420_8bit_YUV.y4m")
    dst_dir_stem = Path("compressed_videos/UVG")
    codecs = ["vp9"]
    levels = [1.5,2.5,4,8]
    for codec in codecs:
        for level in levels:
            meta, _ = transcode_one(src, dst_dir_stem / codec / str(level), codec, level=level)
            print(f"Transcoded {src} to {dst_dir_stem / codec / str(level)} using {codec} at level {level}")
            print(json.dumps(meta, indent=2))
    print("All done!")
    

if __name__ == "__main__":
    #main_only_one()
    main()