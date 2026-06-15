import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.linear_model import LinearRegression

sys.path.append("visualizations")
from vis_vmaf_paradox import load_ti_groups, DATASETS, CODECS, LEVELS
from vis_correlation import collect_points, _ti_lookup

def main():
    ti_groups, bounds = load_ti_groups(DATASETS)
    ti_lookup = _ti_lookup(ti_groups)
    
    points = collect_points(
        ti_lookup=ti_lookup,
        datasets=DATASETS,
        codecs=CODECS,
        levels=LEVELS,
        y_metric="movie_index"
    )
    
    # Aggregate by video
    video_data = {}
    for p in points:
        v = p["video"]
        if v not in video_data:
            video_data[v] = {"vmaf": [], "movie_index": [], "ti_group": p["ti_group"]}
        video_data[v]["vmaf"].append(p["vmaf"])
        video_data[v]["movie_index"].append(p["y"])
        
    avg_points = []
    for v, d in video_data.items():
        avg_points.append({
            "video": v,
            "vmaf": np.mean(d["vmaf"]),
            "y": np.mean(d["movie_index"]),
            "ti_group": d["ti_group"]
        })
    
    print(f"Total videos: {len(avg_points)}")
    
    groups = {1: [], 2: [], 3: [], 4: []}
    for p in avg_points:
        groups[p["ti_group"]].append(p)
        
    # All structured
    struct_vmaf = []
    struct_movie = []
    for g in [1, 2, 4]:
        struct_vmaf.extend([p["vmaf"] for p in groups[g]])
        struct_movie.extend([p["y"] for p in groups[g]])
        
    plcc_str, p_plcc_str = pearsonr(struct_vmaf, struct_movie)
    srocc_str, p_srocc_str = spearmanr(struct_vmaf, struct_movie)
    print(f"Structured Groups (1,2,4): PLCC={plcc_str:.2f} (p={p_plcc_str:.3e}), SROCC={srocc_str:.2f} (p={p_srocc_str:.3e})")
    
    # Group 3
    g3_vmaf = [p["vmaf"] for p in groups[3]]
    g3_movie = [p["y"] for p in groups[3]]
    plcc3, p_plcc3 = pearsonr(g3_vmaf, g3_movie)
    srocc3, p_srocc3 = spearmanr(g3_vmaf, g3_movie)
    print(f"Group 3: PLCC={plcc3:.2f} (p={p_plcc3:.3e}), SROCC={srocc3:.2f} (p={p_srocc3:.3e})")
    
    # ANOVA
    # Fit LR on all videos
    all_vmaf = np.array([p["vmaf"] for p in avg_points]).reshape(-1, 1)
    all_movie = np.array([p["y"] for p in avg_points])
    
    lr = LinearRegression()
    lr.fit(all_vmaf, all_movie)
    
    errors = {1: [], 2: [], 3: [], 4: []}
    for p in avg_points:
        pred = lr.predict([[p["vmaf"]]])[0]
        err = abs(p["y"] - pred)
        errors[p["ti_group"]].append(err)
        
    F, p_val = f_oneway(errors[1], errors[2], errors[3], errors[4])
    N = len(avg_points)
    print(f"ANOVA on absolute errors (global LR): F({3}, {N-4}) = {F:.2f}, p={p_val:.3e}")

if __name__ == "__main__":
    main()
