"""Plot BD-Rate efficiency of AV1 vs H.264 across TI groups.

The chart shows bitrate savings (%) for AV1 when anchored to H.264, averaged
per temporal-information (TI) group. Savings are computed from BD-Rate values
using the VMAF RD curves.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from TI_groups import get_TI_groups


DATASETS = ["UVG", "HEVC_CLASS_B", "BVI-HD"]
ANCHOR = "h264"
CODEC = "av1"
METRIC = "vmaf"
NUM_GROUPS = 4


def _normalize_video_name(name: str) -> str:
	"""Remove codec suffixes and extensions for consistent joins."""
	stem = Path(name).stem
	for suffix in ("_h264", "_hevc", "_vp9", "_av1"):
		if stem.endswith(suffix):
			return stem[: -len(suffix)]
	return stem


def load_ti_groups(datasets, number_of_groups=4):
	ti_groups = get_TI_groups(datasets, number_of_groups)
	records = []
	bounds = []
	for group_id, videos in ti_groups.items():
		if not videos:
			continue
		ti_values = [ti for _, ti in videos]
		min_ti = min(ti_values)
		max_ti = max(ti_values)
		bounds.append(max_ti)
		for video_name, ti_value in videos:
			records.append({
				"dataset": next(ds for ds in datasets if video_name.startswith(ds)),
				"video_base": _normalize_video_name(video_name),
				"ti_value": ti_value,
				"ti_group": group_id,
			})
	ti_df = pd.DataFrame(records)

	return ti_df, bounds


def load_bd_scores(datasets, anchor, codec, metric):
	records = []
	for dataset in datasets:
		bd_path = Path(f"results/bd_scores/bd_scores_{dataset}_{anchor}.json")
		if not bd_path.exists():
			print(f"Missing BD file: {bd_path}")
			continue
		with open(bd_path, "r") as f:
			bd_scores = json.load(f)

		codec_scores = bd_scores.get(codec, {})
		for video_name, metrics in codec_scores.items():
			val = metrics.get(metric)
			if val is None or pd.isna(val):
				continue
			records.append({"dataset": dataset, "video_base": _normalize_video_name(video_name), "bd_metric": float(val)})

	return pd.DataFrame(records)


def _format_group_labels(bounds):
	labels = []
	edges = [-np.inf] + bounds + [np.inf]
	for idx in range(len(edges) - 1):
		low, high = edges[idx], edges[idx + 1]
		if low == -np.inf:
			labels.append(f"G{idx + 1}: <= {high:.1f}")
		elif high == np.inf:
			labels.append(f"G{idx + 1}: >{low:.1f}")
		else:
			labels.append(f"G{idx + 1}: {low:.1f}-{high:.1f}")
	return labels


def plot_bd_savings_by_ti(output_dir="visualizations/bd_rate_by_TI", number_of_groups=NUM_GROUPS):
	ti_df, bounds = load_ti_groups(DATASETS, number_of_groups)
	if ti_df.empty:
		print("No TI data found; aborting plot.")
		return

	bd_df = load_bd_scores(DATASETS, ANCHOR, CODEC, METRIC)
	if bd_df.empty:
		print("No BD scores found; aborting plot.")
		return

	merged = bd_df.merge(ti_df[["dataset", "video_base", "ti_group"]], on=["dataset", "video_base"], how="inner")
	if merged.empty:
		print("No overlap between BD scores and TI groups.")
		return

	merged["bitrate_savings_pct"] = -merged["bd_metric"]

	agg = merged.groupby("ti_group").agg(
		savings_mean=("bitrate_savings_pct", "mean"),
		savings_median=("bitrate_savings_pct", "median"),
		count=("bitrate_savings_pct", "size"),
	).reset_index()

	agg["ti_group_label"] = agg["ti_group"].apply(int).astype(str)
	group_labels = _format_group_labels(bounds)
	label_map = {str(idx + 1): label for idx, label in enumerate(group_labels)}
	agg["ti_group_desc"] = agg["ti_group_label"].map(label_map)

	palette = sns.color_palette("crest", n_colors=len(agg))
	plt.figure(figsize=(8, 5))
	ax = sns.barplot(data=agg, x="ti_group_label", y="savings_mean", palette=palette, width=0.55)
	ax.axhline(0, color="0.3", linewidth=1)
	ax.set_xlabel("TI group")
	ax.set_ylabel("Bitrate savings vs H.264 (%, VMAF)")
	ax.set_title("AV1 BD-Rate efficiency by TI group")
	ax.set_xticklabels([label_map[label] for label in agg["ti_group_label"]])

	for bar, value in zip(ax.containers[0], agg["savings_mean"]):
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height(),
			f"{value:.1f}%",
			ha="center",
			va="bottom",
		)

	plt.tight_layout()
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	outfile = output_dir / "bd_rate_av1_vs_h264_by_ti.png"
	plt.savefig(outfile, dpi=300)
	plt.close()
	print(f"Saved plot to {outfile}")


if __name__ == "__main__":
	plot_bd_savings_by_ti()
