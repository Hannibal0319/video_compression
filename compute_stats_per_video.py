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
    
    # Group points by video
    video_data = {}
    for p in points:
        v = p["video"]
        if v not in video_data:
            video_data[v] = {"vmaf": [], "movie_index": [], "ti_group": p["ti_group"]}
        video_data[v]["vmaf"].append(p["vmaf"])
        video_data[v]["movie_index"].append(p["y"])
        
    group_plcc = {1: [], 2: [], 3: [], 4: []}
    group_srocc = {1: [], 2: [], 3: [], 4: []}
    
    video_errors = {1: [], 2: [], 3: [], 4: []}
    
    # To compute error distribution, let's fit a linear model per video and take the mean absolute error, or variance of errors?
    # "mean error variance of the unpredictable group"
    for v, d in video_data.items():
        vmaf = np.array(d["vmaf"])
        movie = np.array(d["movie_index"])
        if len(vmaf) < 2: continue
        
        r, p_plcc = pearsonr(vmaf, movie)
        rho, p_srocc = spearmanr(vmaf, movie)
        
        g = d["ti_group"]
        group_plcc[g].append((r, p_plcc))
        group_srocc[g].append((rho, p_srocc))
        
        # Fit simple LR to get errors
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(vmaf.reshape(-1, 1), movie)
        preds = lr.predict(vmaf.reshape(-1, 1))
        # Variance of errors (MSE)
        mse = np.mean((movie - preds)**2)
        video_errors[g].append(mse)

    # Aggregated stats
    struct_plcc = []
    struct_srocc = []
    for g in [1, 2, 4]:
        struct_plcc.extend([x[0] for x in group_plcc[g]])
        struct_srocc.extend([x[0] for x in group_srocc[g]])
        
    print(f"Structured Groups (1,2,4) - Mean PLCC: {np.mean(struct_plcc):.2f}, Mean SROCC: {np.mean(struct_srocc):.2f}")
    
    g3_plcc = [x[0] for x in group_plcc[3]]
    g3_srocc = [x[0] for x in group_srocc[3]]
    print(f"Group 3 - Mean PLCC: {np.mean(g3_plcc):.2f}, Mean SROCC: {np.mean(g3_srocc):.2f}")
    
    # ANOVA on error variance
    F, p_val = f_oneway(video_errors[1], video_errors[2], video_errors[3], video_errors[4])
    N = sum(len(video_errors[g]) for g in [1, 2, 3, 4])
    print(f"ANOVA on prediction error variance (MSE per video): F({3}, {N-4}) = {F:.2f}, p={p_val:.3e}")

if __name__ == "__main__":
    main()
