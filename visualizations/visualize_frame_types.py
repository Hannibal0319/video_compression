import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from TI_groups import get_TI_groups


datasets = ["UVG","HEVC_CLASS_B","BVI-HD"]
levels = [1,1.5, 2, 2.5, 3,4,8]
codecs = ["h264", "hevc", "vp9","av1"]

def visualize_frame_type_distribution(output_dir='visualizations/frame_type_distribution'):
    """
    Scans for frame type analysis JSON files, aggregates the data,
    and generates stacked bar charts to visualize frame distribution.
    """

    results_dir = Path('results/frame_types')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    all_data = []
    json_files = list(results_dir.rglob('*.json'))
    print(f"Found {len(json_files)} JSON files to process.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract metadata from path
            parts = file_path.parts
            # Expected path: .../results/frame_types/DATASET/CODEC/LEVEL/FILENAME.json
            dataset = parts[-4]
            codec = parts[-3]
            level = parts[-2]
            
            percentages = data.get('frame_percentages', {})
            record = {
                'dataset': dataset,
                'codec': codec,
                'level': level,
                'I': percentages.get('I', 0),
                'P': percentages.get('P', 0),
                'B': percentages.get('B', 0)
            }
            all_data.append(record)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Could not process file {file_path}: {e}")
            continue

    if not all_data:
        print("No data to visualize.")
        return

    df = pd.DataFrame(all_data)
    
    # Convert level to a numeric type for proper sorting
    df['level'] = pd.to_numeric(df['level'], errors='coerce')
    df.dropna(subset=['level'], inplace=True)

    # Group by dataset, codec, and level to get average percentages
    agg_df = df.groupby(['dataset', 'codec', 'level']).mean().reset_index()

    # Create a plot for each dataset
    for dataset_name, group_df in agg_df.groupby('dataset'):
        print(f"Generating plot for dataset: {dataset_name}")
        
        pivot_df = group_df.pivot(index=['codec', 'level'], columns=[], values=['I', 'P', 'B'])
        pivot_df.sort_index(level='level', inplace=True)

        #plot I, P, B frame distributions at different levels do one for each codec
        for codec in codecs:
            codec_df = pivot_df.loc[codec]
            fig, ax = plt.subplots(figsize=(8, 6))
            bottom = np.zeros(len(codec_df))
            levels_sorted = sorted(codec_df.index.get_level_values('level').unique())
            for frame_type, color in zip(['I', 'P', 'B'], ['#ff9999','#66b3ff','#99ff99']):
                values = []
                for level in levels_sorted:
                    try:
                        val = codec_df.loc[level][frame_type]
                    except KeyError:
                        val = 0
                    values.append(val)
                ax.bar(
                    range(len(values)),
                    values,
                    bottom=bottom,
                    label=frame_type,
                    color=color
                )
                bottom += np.array(values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(
                [f"L{level}" for level in levels_sorted],
                rotation=45,
                ha='right'
            )
            ax.set_ylabel('Average Frame Percentage (%)')
            ax.set_title(f'Frame Type Distribution for {codec.upper()} on {dataset_name} Dataset')
            ax.legend(title='Frame Types')
            #plt.tight_layout()
            plot_file = output_dir / f'frame_type_distribution_{dataset_name}_{codec}.png'
            plt.savefig(plot_file)
            plt.close()

def visualize_frame_type_distribution_by_TI_groups(output_dir='visualizations/frame_type_distribution_by_TI'):
    """
    Visualizes frame type distribution categorized by TI groups.
    """

    results_dir = Path('results/frame_types')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    TI_groups = get_TI_groups(datasets=datasets,number_of_groups=6)
    print(f"Loaded TI groups for datasets: {list(TI_groups.keys())}")
    # output: [1, 2, 3, 4]
    print(f"Sample TI groups: {list(TI_groups.items())[:5]}")
    all_data = []
    
    for group_id,lista in TI_groups.items():
        try:
            for video_path,_ in lista:
                for codec in codecs:
                    for level in levels:
                        # get json path
                        video_path = Path(video_path)
                        
                        try:
                            json_path = results_dir / "UVG" /video_path.parent.name / codec / str(level) / f"{video_path.stem}_{codec}.json"
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError) as e:
                            try:
                                json_path = results_dir / "HEVC_CLASS_B" /video_path.parent.name / codec / str(level) / f"{video_path.stem}_{codec}.json"
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                            except (FileNotFoundError, json.JSONDecodeError) as e:
                                print(json_path, e)
                                continue

                        # Extract metadata from path
                        
                        
                        percentages = data.get('frame_percentages', {})

                        record = {
                            'codec': codec,
                            'level': level,
                            'TI_group': group_id,
                            'I': percentages.get('I', 0),
                            'P': percentages.get('P', 0),
                            'B': percentages.get('B', 0)
                        }
                        all_data.append(record)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Could not process file {json_path}: {e}")
            continue

    if not all_data:
        print("No data to visualize.")
        return

    df = pd.DataFrame(all_data)
    
    # Convert level to a numeric type for proper sorting
    df['level'] = pd.to_numeric(df['level'], errors='coerce')
    df.dropna(subset=['level'], inplace=True)

    # Group by dataset, codec, level, and TI_group to get average percentages
    agg_df = df.groupby([ 'codec', 'level', 'TI_group']).mean().reset_index()
    print(agg_df.head())
    # Create a plot for each TI group
    for TI_group_name, group_df in agg_df.groupby('TI_group'):
        print(f"Generating plot for TI group: {TI_group_name}")
        
        pivot_df = group_df.pivot(index=['codec', 'level'], columns=[], values=['I', 'P', 'B'])
        pivot_df.sort_index(level='level', inplace=True)

        # Plot I, P, B frame distributions at different levels for each codec
        for codec in codecs:
            codec_df = pivot_df.loc[codec]
            fig, ax = plt.subplots(figsize=(8, 6))
            bottom = np.zeros(len(codec_df))
            levels_sorted = sorted(codec_df.index.get_level_values('level').unique())
            for frame_type, color in zip(['I', 'P', 'B'], ['#ff9999','#66b3ff','#99ff99']):
                values = []
                for level in levels_sorted:
                    try:
                        val = codec_df.loc[level][frame_type]
                    except KeyError:
                        val = 0
                    values.append(val)
                print(f"TI Group: {TI_group_name}, Codec: {codec}, Level: {level}, Frame Type: {frame_type}, Values: {values}")
                ax.bar(
                    range(len(values)),
                    values,
                    bottom=bottom,
                    label=frame_type,
                    color=color
                )
                bottom += np.array(values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(
                [f"L{level}" for level in levels_sorted],
                rotation=45,
                ha='right'
            )
            ax.set_ylabel('Average Frame Percentage (%)')
            ax.set_title(f'Frame Type Distribution for {codec.upper()} - TI Group: {TI_group_name}')
            ax.legend(title='Frame Types')
            #plt.tight_layout()
            plot_file = output_dir / f'frame_type_distribution_TI_{TI_group_name}_{codec}.png'
            plt.savefig(plot_file)
            plt.close()
    print("Visualization complete.")
    #calculate groups averages for I-frames by codec
    ti_i_frame_stats = agg_df.groupby(['TI_group'])['I'].mean().reset_index()
    ti_i_frame_stats = ti_i_frame_stats.sort_values(by='I').reset_index(drop=True)
    
    print("TI Group I-frame averages:")
    print(ti_i_frame_stats)
    # visualize as bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        ti_i_frame_stats['TI_group'].astype(str),
        ti_i_frame_stats['I']*100,
        color='#ff9999'
    )
    ax.set_xlabel('TI Group')
    ax.set_ylabel('Average I Frame Percentage (%)')
    ax.set_title('Average I Frame Percentage by TI Group')
    plot_file = output_dir / f'average_I_frame_percentage_by_TI_group.png'
    plt.savefig(plot_file)
    plt.close()
    
if __name__ == '__main__':
    visualize_frame_type_distribution_by_TI_groups()
    #visualize_frame_type_distribution()
    pass
