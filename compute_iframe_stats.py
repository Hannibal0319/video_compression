import json
import pandas as pd
from pathlib import Path

import sys
sys.path.append("visualizations")
from TI_groups import get_TI_groups

datasets = ["UVG", "HEVC_CLASS_B", "BVI-HD"]
levels = [1, 1.5, 2, 2.5, 3, 4, 8]
codecs = ["h264", "hevc", "vp9", "av1"]

def main():
    results_dir = Path('results/frame_types')
    TI_groups = get_TI_groups(datasets=datasets, number_of_groups=4)
    
    all_data = []
    
    # We will search by matching video_stem
    json_files = list(results_dir.rglob('*.json'))
    
    for group_id, lista in TI_groups.items():
        for video_path, _ in lista:
            stem = Path(video_path).stem
            for codec in codecs:
                for level in levels:
                    # Find matching file manually since the directory structure in the previous script had typos
                    matched_file = None
                    for jf in json_files:
                        if codec in jf.parts and str(level) in jf.parts and stem in jf.name:
                            matched_file = jf
                            break
                    if matched_file:
                        with open(matched_file, 'r') as f:
                            data = json.load(f)
                        percentages = data.get('frame_percentages', {})
                        record = {
                            'codec': codec,
                            'level': level,
                            'TI_group': group_id,
                            'I': percentages.get('I', 0)
                        }
                        all_data.append(record)

    df = pd.DataFrame(all_data)
    
    agg_df = df.groupby(['TI_group', 'codec'])['I'].mean().unstack()
    agg_df['Group Average'] = df.groupby('TI_group')['I'].mean()
    
    print("TI Group | H.264 | VP9 | HEVC | AV1 | Group Average")
    print("---|---|---|---|---|---")
    
    group_labels = {
        1: "Group 1 (Static)",
        2: "Group 2 (Predictable)",
        3: "Group 3 (Unpredictable)",
        4: "Group 4 (Global)"
    }
    
    for g in [1, 2, 3, 4]:
        h264 = agg_df.loc[g, 'h264']
        vp9 = agg_df.loc[g, 'vp9']
        hevc = agg_df.loc[g, 'hevc']
        av1 = agg_df.loc[g, 'av1']
        avg = agg_df.loc[g, 'Group Average']
        
        label = group_labels.get(g, f"Group {g}")
        print(f"{label} | {h264:.3f}% | {vp9:.3f}% | {hevc:.3f}% | {av1:.3f}% | {avg:.4f}%")

if __name__ == '__main__':
    main()
