import argparse
import json
import os
import subprocess
from pathlib import Path
import concurrent.futures
from collections import Counter

def run_command(cmd):
    """Runs a command and returns its output."""
    try:
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        return e.output

def analyze_video_frames(video_path):
    """
    Analyzes a video file to get frame type counts using ffprobe.
    """
    print(f"Analyzing {video_path}...")
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0',
        str(video_path)
    ]

    output = run_command(cmd)
    if not output:
        return {}

    # Clean up each line to remove potential commas or extra characters
    frame_types = [line.strip().replace(',', '') for line in output.strip().split('\n')]
    
    counts = Counter(frame_types)
    total_frames = len(frame_types)

    stats = {
        "total_frames": total_frames,
        "frame_counts": dict(counts),
        "frame_percentages": {k: (v / total_frames) * 100 for k, v in counts.items()}
    }
    
    return stats

def process_file(video_path, output_dir):
    """Processes a single video file and saves the analysis results."""
    try:
        frame_stats = analyze_video_frames(video_path)
        if not frame_stats:
            print(f"No frame stats extracted for {video_path}")
            return

        # Use the video's relative path to structure the output
        relative_path = video_path.relative_to(Path('compressed_videos'))
        output_file = output_dir / relative_path.with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(frame_stats, f, indent=4)
        
        print(f"Saved analysis for {video_path} to {output_file}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze video files for frame type distribution.")
    parser.add_argument('--input_dir', type=str, default='compressed_videos', help='Directory containing compressed videos.')
    parser.add_argument('--output_dir', type=str, default='results/frame_types', help='Directory to save analysis results.')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1, help='Number of worker processes.')
    
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    video_extensions = ['.mp4', '.mkv', '.webm', '.avi']
    video_files = [p for p in input_path.rglob('*') if p.suffix.lower() in video_extensions]
    print(f"Found {len(video_files)} video files to process.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, video_file, output_path) for video_file in video_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A task generated an exception: {e}")
    

if __name__ == '__main__':
    main()
