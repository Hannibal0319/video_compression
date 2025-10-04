import argparse, concurrent.futures as fut, json, os, shlex, subprocess, sys, time
from pathlib import Path

crfs = [i for i in range(19, 40)]  # CRF values from 19 to 39

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
#    "av1": {
#        "ext": "mkv",
#        "args": '-c:v libaom-av1 -preset 6 -c:a libopus -b:a 128k'
#    },
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

def transcode_one(src, dst_dir, codec, vmaf_ref=None):
    prof = CODEC_PROFILES[codec]
    out_path = (dst_dir / (src.stem + f'_{codec}.' + prof["ext"])).absolute()
    os.makedirs(dst_dir, exist_ok=True)
    src_abs = src.absolute()
    level = 2
    level_params = compression_level_params(level)
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
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--codecs","-c", nargs="+", default=["h264" ,"hevc","av1","vp9"], choices=CODEC_PROFILES.keys())
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--vmaf", action="store_true", help="Compute VMAF vs original")
    args = ap.parse_args()
    
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    videos = [p for p in in_dir.rglob("*") if p.suffix.lower() in {".mp4",".mov",".mkv",".mxf",".avi",".webm"}]
    if not videos:
        print("No input videos found.", file=sys.stderr)
        sys.exit(2)

    tasks = []
    print(f"Transcoding {len(videos)} videos to {args.codecs} using {args.workers} workers...")
    with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for src in videos:
            dst_sub = out_dir / src.relative_to(in_dir).parent
            for codec in args.codecs:
                tasks.append(ex.submit(transcode_one, src, dst_sub, codec, vmaf_ref=str(src) if args.vmaf else None))
        for t in fut.as_completed(tasks):
            meta, _ = t.result()
            print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
    # videos\Jockey_1920x1080_120fps_420_8bit_YUV.mp4