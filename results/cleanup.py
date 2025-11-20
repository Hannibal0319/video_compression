import json
import os

datasets = ["UVG","HEVC_CLASS_B"]
codecs = ["h264","hevc","vp9"]
levels = ["1","1.5","2","2.5","3","4","8"]

def cleanup_unwanted_entries():
    for dataset in datasets:
        for codec in codecs:
            for level in levels:
                result_file = f"results/eval_metrics_{dataset}_{codec}_level{level}.json"
                if os.path.exists(result_file):
                    data = json.load(open(result_file, "r"))
                    cleaned_data = {}
                    for video, _ in data.items():
                        if video.split("_")[-1].split(".")[0] == codec:
                            cleaned_data[video] = data[video]
                    with open(result_file, "w") as f:
                        json.dump(cleaned_data, f, indent=4)
                    print(f"Cleaned up {result_file}, kept {len(cleaned_data)}/{len(data)} entries.")

if __name__ == "__main__":
    #cleanup_unwanted_entries()
    pass