import json

def get_TI_groups(datasets):
    all_ti_data = {}
    for dataset in datasets:
        with open(f"results/eval_metrics_{dataset}_TI.json", 'r') as f:
            ti_per_video = json.load(f)
        for video_name, ti_data in ti_per_video.items():
            print(f"Video: {video_name}, TI: {ti_data}")
            all_ti_data[video_name] = ti_data
    # Now all_ti_data contains TI values for all videos across datasets

    #make 4 groups based on TI values so approx equal number of videos in each group
    ti_values = list(all_ti_data.values())
    ti_values.sort()
    n = len(ti_values)
    group_bounds = [
        ti_values[n // 4],
        ti_values[n // 2],
        ti_values[3 * n // 4]
    ]
    ti_groups = {1: [], 2: [], 3: [], 4: []}
    for video_name, ti_value in all_ti_data.items():
        if ti_value <= group_bounds[0]:
            ti_groups[1].append((video_name, ti_value))
        elif ti_value <= group_bounds[1]:
            ti_groups[2].append((video_name, ti_value))
        elif ti_value <= group_bounds[2]:
            ti_groups[3].append((video_name, ti_value))
        else:
            ti_groups[4].append((video_name, ti_value))
    
    return ti_groups

if __name__ == "__main__":
    datasets = ["UVG","HEVC_CLASS_B"]
    ti_groups = get_TI_groups(datasets)
    for group_id, videos in ti_groups.items():
        print(f"TI Group {group_id}:")
        for video_name, ti_value in videos:
            print(f"  {video_name}: TI={ti_value}")