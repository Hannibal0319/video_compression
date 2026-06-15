import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.linear_model import LinearRegression

sys.path.append("visualizations")
from vis_vmaf_paradox import load_ti_groups, DATASETS, CODECS, LEVELS
from vis_correlation import collect_points, _ti_lookup

def get_stats():
    ti_groups, bounds = load_ti_groups(DATASETS)
    ti_lookup = _ti_lookup(ti_groups)
    
    points = collect_points(
        ti_lookup=ti_lookup,
        datasets=DATASETS,
        codecs=CODECS,
        levels=LEVELS,
        y_metric="movie_index"
    )

    # Strategy 1: Per-video correlation (across 32 points: 4 codecs x 8 levels)
    # The reviewer said "For structured sequences... metrics scale predictably... For TI Group 3, this linear bond completely breaks down."
    # Let's calculate the correlation per video, then average the correlation coefficients for the groups.
    video_data = {}
    for p in points:
        v = p["video"]
        if v not in video_data:
            video_data[v] = {"vmaf": [], "movie_index": [], "ti_group": p["ti_group"]}
        video_data[v]["vmaf"].append(p["vmaf"])
        video_data[v]["movie_index"].append(p["y"])
        
    plccs = {1: [], 2: [], 3: [], 4: []}
    sroccs = {1: [], 2: [], 3: [], 4: []}
    mse_errors = {1: [], 2: [], 3: [], 4: []}
    
    for v, d in video_data.items():
        vmaf = np.array(d["vmaf"])
        movie = np.array(d["movie_index"])
        
        if len(vmaf) < 2: continue
        
        r, _ = pearsonr(vmaf, movie)
        rho, _ = spearmanr(vmaf, movie)
        g = d["ti_group"]
        
        plccs[g].append(abs(r))
        sroccs[g].append(abs(rho))
        
        # Linear regression to predict MOVIE from VMAF
        lr = LinearRegression()
        lr.fit(vmaf.reshape(-1, 1), movie)
        preds = lr.predict(vmaf.reshape(-1, 1))
        # use normalized MSE as variance, or standard error of regression
        # if the linear bond breaks down, the residual variance will be higher.
        res_var = np.var(movie - preds) / (np.var(movie) + 1e-9) # R-squared is 1 - this. So this is 1 - R^2.
        mse_errors[g].append(res_var)

    struct_plcc = []
    struct_srocc = []
    for g in [1, 2, 4]:
        struct_plcc.extend(plccs[g])
        struct_srocc.extend(sroccs[g])
        
    g3_plcc = plccs[3]
    g3_srocc = sroccs[3]
    
    print("=== Strategy 1: Average per-video correlation across codecs and bitrates ===")
    print(f"Structured Groups: |PLCC| = {np.mean(struct_plcc):.3f}, |SROCC| = {np.mean(struct_srocc):.3f}")
    print(f"Group 3          : |PLCC| = {np.mean(g3_plcc):.3f}, |SROCC| = {np.mean(g3_srocc):.3f}")
    
    F, p = f_oneway(mse_errors[1], mse_errors[2], mse_errors[3], mse_errors[4])
    N = sum(len(mse_errors[g]) for g in [1, 2, 3, 4])
    print(f"ANOVA on normalized residual variance (per video): F({3}, {N-4}) = {F:.2f}, p = {p:.3e}")
    print()

    # Strategy 2: Pooled points but normalized per video (Z-score normalization)
    norm_groups = {1: {"vmaf": [], "movie": []}, 2: {"vmaf": [], "movie": []}, 3: {"vmaf": [], "movie": []}, 4: {"vmaf": [], "movie": []}}
    
    for v, d in video_data.items():
        vmaf = np.array(d["vmaf"])
        movie = np.array(d["movie_index"])
        
        if len(vmaf) < 2 or np.std(vmaf) == 0 or np.std(movie) == 0: continue
        
        vmaf_z = (vmaf - np.mean(vmaf)) / np.std(vmaf)
        movie_z = (movie - np.mean(movie)) / np.std(movie)
        
        g = d["ti_group"]
        norm_groups[g]["vmaf"].extend(vmaf_z)
        norm_groups[g]["movie"].extend(movie_z)

    # aggregate structured
    struct_vmaf = []
    struct_movie = []
    for g in [1, 2, 4]:
        struct_vmaf.extend(norm_groups[g]["vmaf"])
        struct_movie.extend(norm_groups[g]["movie"])
        
    r_struct, p_struct = pearsonr(struct_vmaf, struct_movie)
    rho_struct, _ = spearmanr(struct_vmaf, struct_movie)
    
    r_g3, p_g3 = pearsonr(norm_groups[3]["vmaf"], norm_groups[3]["movie"])
    rho_g3, _ = spearmanr(norm_groups[3]["vmaf"], norm_groups[3]["movie"])
    
    print("=== Strategy 2: Pooled Z-score normalized correlation ===")
    print(f"Structured Groups: |PLCC| = {abs(r_struct):.3f} (p={p_struct:.3e}), |SROCC| = {abs(rho_struct):.3f}")
    print(f"Group 3          : |PLCC| = {abs(r_g3):.3f} (p={p_g3:.3e}), |SROCC| = {abs(rho_g3):.3f}")
    
    # ANOVA on absolute errors of global linear regression on normalized data
    lr_struct = LinearRegression().fit(np.array(struct_vmaf).reshape(-1, 1), struct_movie)
    lr_g3 = LinearRegression().fit(np.array(norm_groups[3]["vmaf"]).reshape(-1, 1), norm_groups[3]["movie"])
    
    all_vmaf_norm = np.concatenate([struct_vmaf, norm_groups[3]["vmaf"]])
    all_movie_norm = np.concatenate([struct_movie, norm_groups[3]["movie"]])
    lr_global = LinearRegression().fit(all_vmaf_norm.reshape(-1, 1), all_movie_norm)
    
    errs = {1: [], 2: [], 3: [], 4: []}
    for g in [1, 2, 3, 4]:
        vmaf_arr = np.array(norm_groups[g]["vmaf"]).reshape(-1, 1)
        movie_arr = np.array(norm_groups[g]["movie"])
        if len(vmaf_arr) > 0:
            preds = lr_global.predict(vmaf_arr)
            errs[g] = np.abs(movie_arr - preds)
            
    F2, p2 = f_oneway(errs[1], errs[2], errs[3], errs[4])
    N2 = sum(len(errs[g]) for g in [1, 2, 3, 4])
    print(f"ANOVA on absolute prediction errors (Z-score pooled): F({3}, {N2-4}) = {F2:.2f}, p = {p2:.3e}")

get_stats()
