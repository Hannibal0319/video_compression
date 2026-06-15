import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway

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
    
    # We want to group by (video, codec) to find the scaling behavior over bitrates
    video_codec_data = {}
    for p in points:
        k = (p["video"], p["codec"])
        if k not in video_codec_data:
            video_codec_data[k] = {"vmaf": [], "movie_index": [], "ti_group": p["ti_group"]}
        video_codec_data[k]["vmaf"].append(p["vmaf"])
        video_codec_data[k]["movie_index"].append(p["y"])
        
    group_plcc = {1: [], 2: [], 3: [], 4: []}
    group_srocc = {1: [], 2: [], 3: [], 4: []}
    video_errors = {1: [], 2: [], 3: [], 4: []}
    
    for k, d in video_codec_data.items():
        vmaf = np.array(d["vmaf"])
        movie = np.array(d["movie_index"])
        if len(vmaf) < 3: continue
        
        r, p_plcc = pearsonr(vmaf, movie)
        rho, p_srocc = spearmanr(vmaf, movie)
        
        g = d["ti_group"]
        group_plcc[g].append((r, p_plcc))
        group_srocc[g].append((rho, p_srocc))
        
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(vmaf.reshape(-1, 1), movie)
        preds = lr.predict(vmaf.reshape(-1, 1))
        mse = np.var(movie - preds)
        video_errors[g].append(mse)

    # Aggregated
    struct_plcc = []
    struct_srocc = []
    struct_p = []
    for g in [1, 2, 4]:
        struct_plcc.extend([abs(x[0]) for x in group_plcc[g]])
        struct_srocc.extend([abs(x[0]) for x in group_srocc[g]])
        struct_p.extend([x[1] for x in group_plcc[g]])
        
    g3_plcc = [abs(x[0]) for x in group_plcc[3]]
    g3_srocc = [abs(x[0]) for x in group_srocc[3]]
    g3_p = [x[1] for x in group_plcc[3]]
    
    print(f"Structured Groups - Mean |PLCC|: {np.mean(struct_plcc):.2f}, Median p-val: {np.median(struct_p):.3e}")
    print(f"Group 3 - Mean |PLCC|: {np.mean(g3_plcc):.2f}, Median p-val: {np.median(g3_p):.3e}")
    
    # ANOVA
    F, p_val = f_oneway(video_errors[1], video_errors[2], video_errors[3], video_errors[4])
    N = sum(len(video_errors[g]) for g in [1, 2, 3, 4])
    print(f"ANOVA on prediction error variance (MSE per video-codec): F({3}, {N-4}) = {F:.2f}, p={p_val:.3e}")

if __name__ == "__main__":
    main()
