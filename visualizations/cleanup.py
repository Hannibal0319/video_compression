import os
import glob

def cleanup_plots_by_video():
    """
    Remove all plot files generated per video to clean up the directory.
    """
    plot_patterns = [
        "visualizations/plots_by_video/*.png"
    ]
    for pattern in plot_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {e}")

if __name__ == "__main__":
    cleanup_plots_by_video()