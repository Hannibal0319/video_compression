import json

def get_TI_groups(datasets,number_of_groups=4):
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
    # Determine group bounds so every group gets approx equal number of videos
    group_bounds = [
        ti_values[min((i + 1) * n // number_of_groups - 1, n - 1)] for i in range(number_of_groups - 1)
    ]
    print(f"TI group bounds: {group_bounds}")
    ti_groups = {i: [] for i in range(1, number_of_groups + 1)}
    for video_name, ti_value in all_ti_data.items():
        for i, bound in enumerate(group_bounds):
            if ti_value <= bound:
                ti_groups[i + 1].append((video_name, ti_value))
                break
        else:
            ti_groups[number_of_groups].append((video_name, ti_value))
    
    print("TI Groups:")
    for group_id, videos in ti_groups.items():
        print(f"Group {group_id}:")
        for video_name, ti_value in videos:
            print(f"  {video_name}: TI={ti_value}")
    return ti_groups

if __name__ == "__main__":
    datasets = ["UVG","HEVC_CLASS_B"]
    ti_groups = get_TI_groups(datasets)
    for group_id, videos in ti_groups.items():
        print(f"TI Group {group_id}:")
        for video_name, ti_value in videos:
            print(f"  {video_name}: TI={ti_value}")