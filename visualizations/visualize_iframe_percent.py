import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


DATASETS = ["UVG", "HEVC_CLASS_B", "BVI-HD"]
CODECS = ["h264", "hevc", "vp9", "av1"]


def _normalize_video_name(name: str) -> str:
	"""Strip container/codec suffixes so TI and frame-type keys match."""
	stem = Path(name).stem
	for codec in CODECS:
		suffix = f"_{codec}"
		if stem.endswith(suffix):
			stem = stem[: -len(suffix)]
			break
	return stem


def _load_ti_groups(datasets, number_of_groups=4):
	"""Load TI values and assign group ids using the existing binning rule."""

	rows = []
	ti_values = []
	for dataset in datasets:
		ti_path = Path(f"results/eval_metrics_{dataset}_TI.json")
		if not ti_path.exists():
			print(f"Missing TI file: {ti_path}")
			continue
		with open(ti_path, "r") as f:
			ti_per_video = json.load(f)
		for video_name, ti in ti_per_video.items():
			rows.append({"dataset": dataset, "video_base": _normalize_video_name(video_name), "ti": ti})
			ti_values.append(ti)

	if not rows:
		return pd.DataFrame(), []

	ti_values.sort()
	n = len(ti_values)
	bounds = [ti_values[min((i + 1) * n // number_of_groups - 1, n - 1)] for i in range(number_of_groups - 1)]

	def assign_group(ti_value: float) -> int:
		for idx, bound in enumerate(bounds):
			if ti_value <= bound:
				return idx + 1
		return number_of_groups

	ti_df = pd.DataFrame(rows)
	ti_df["ti_group"] = ti_df["ti"].apply(assign_group)
	return ti_df, bounds


def _load_frame_type_percentages():
	"""Load frame type percentages for every dataset/codec/level/video."""

	base_dir = Path("results/frame_types")
	if not base_dir.exists():
		print(f"Frame type directory not found: {base_dir}")
		return pd.DataFrame()

	records = []
	for file_path in base_dir.rglob("*.json"):
		try:
			dataset, codec, level = file_path.parts[-4], file_path.parts[-3], file_path.parts[-2]
		except ValueError:
			continue

		try:
			with open(file_path, "r") as f:
				data = json.load(f)
		except (json.JSONDecodeError, OSError):
			continue

		percentages = data.get("frame_percentages", {})
		counts = data.get("frame_counts", {})
		total_frames = data.get("total_frames", 0)
		records.append(
			{
				"dataset": dataset,
				"codec": codec,
				"level": level,
				"video_base": _normalize_video_name(file_path.stem),
				"I": percentages.get("I", 0.0),
				"P": percentages.get("P", 0.0),
				"B": percentages.get("B", 0.0),
				"I_count": counts.get("I", 0),
				"total_frames": total_frames,
			}
		)

	if not records:
		return pd.DataFrame()

	df = pd.DataFrame(records)
	df["level"] = pd.to_numeric(df["level"], errors="coerce")
	df.dropna(subset=["level"], inplace=True)
	return df


def plot_i_frame_percentage_by_ti_group(output_dir="visualizations/frame_type_distribution_by_TI", number_of_groups=4):
	"""Create grouped bars of I-frame percentage per codec for each TI group (frame-weighted)."""

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	ti_df, bounds = _load_ti_groups(DATASETS, number_of_groups)
	if ti_df.empty:
		print("No TI data found; skipping plot.")
		return

	frame_df = _load_frame_type_percentages()
	if frame_df.empty:
		print("No frame type data found; skipping plot.")
		return

	merged = frame_df.merge(ti_df[["dataset", "video_base", "ti_group"]], on=["dataset", "video_base"], how="inner")
	if merged.empty:
		print("No overlap between TI data and frame type data.")
		return

	# Frame-weighted aggregation: sum I-frame counts / total frames per codec within each TI group.
	agg = merged.groupby(["ti_group", "codec"], as_index=False).agg({"I_count": "sum", "total_frames": "sum"})
	agg["I_pct"] = np.where(agg["total_frames"] > 0, 100 * agg["I_count"] / agg["total_frames"], 0.0)

	# Also compute overall (all codecs) per TI group for table-style comparison.
	overall = merged.groupby(["ti_group"], as_index=False).agg({"I_count": "sum", "total_frames": "sum"})
	overall["I_pct"] = np.where(overall["total_frames"] > 0, 100 * overall["I_count"] / overall["total_frames"], 0.0)
	print("Overall I-frame % by TI group (frame-weighted across codecs/levels):")
	print(overall[["ti_group", "I_pct"]])

	agg["ti_group"] = agg["ti_group"].astype(str)
	codec_order = CODECS
	palette = {
		"h264": "#1f77b4",
		"hevc": "#ff7f0e",
		"vp9": "#2ca02c",
		"av1": "#d62728",
	}

	plt.figure(figsize=(8, 6))
	ax = sns.barplot(
		data=agg,
		x="ti_group",
		y="I_pct",
		hue="codec",
		hue_order=codec_order,
		palette=palette,
		width=0.5,
	)
	ax.set_xlabel("TI group")
	ax.set_ylabel("I-frame percentage (frame-weighted, %)")
	ax.set_title("I-frame share by codec across TI groups")

	for container in ax.containers:
		ax.bar_label(container, fmt="%.2f", padding=2)

	ax.legend(title="Codec", loc="upper right", fontsize=10)
	plt.tight_layout()

	outfile = output_path / "i_frame_percentage_by_ti_group.png"
	plt.savefig(outfile, dpi=300)
	plt.close()
	print(f"Saved plot to {outfile}")


def plot_i_frame_percentage_by_ti_group_unweighted(output_dir="visualizations/frame_type_distribution_by_TI_unweighted", number_of_groups=4):
	"""Create grouped bars of average I-frame percentage per codec for each TI group (simple mean per video)."""

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	ti_df, bounds = _load_ti_groups(DATASETS, number_of_groups)
	if ti_df.empty:
		print("No TI data found; skipping plot.")
		return

	frame_df = _load_frame_type_percentages()
	if frame_df.empty:
		print("No frame type data found; skipping plot.")
		return

	merged = frame_df.merge(ti_df[["dataset", "video_base", "ti_group"]], on=["dataset", "video_base"], how="inner")
	if merged.empty:
		print("No overlap between TI data and frame type data.")
		return

	# Unweighted aggregation: mean of I-frame percentages across videos per codec within each TI group.
	agg = merged.groupby(["ti_group", "codec"], as_index=False)["I"].mean()
	agg.rename(columns={"I": "I_pct"}, inplace=True)

	# Also compute overall (all codecs) per TI group for quick comparison.
	overall = merged.groupby(["ti_group"], as_index=False)["I"].mean()
	overall.rename(columns={"I": "I_pct"}, inplace=True)
	print("Overall I-frame % by TI group (unweighted mean across videos/levels/codecs):")
	print(overall[["ti_group", "I_pct"]])

	agg["ti_group"] = agg["ti_group"].astype(str)
	codec_order = CODECS
	palette = {
		"h264": "#1f77b4",
		"hevc": "#ff7f0e",
		"vp9": "#2ca02c",
		"av1": "#d62728",
	}

	plt.figure(figsize=(8, 6))
	ax = sns.barplot(
		data=agg,
		x="ti_group",
		y="I_pct",
		hue="codec",
		hue_order=codec_order,
		palette=palette,
		width=0.5,
	)
	ax.set_xlabel("TI group")
	ax.set_ylabel("I-frame percentage (unweighted mean, %)")
	ax.set_title("I-frame share by codec across TI groups (unweighted per video)")

	for container in ax.containers:
		ax.bar_label(container, fmt="%.2f", padding=2)

	ax.legend(title="Codec", loc="upper right", fontsize=10)
	plt.tight_layout()

	outfile = output_path / "i_frame_percentage_by_ti_group_unweighted.png"
	plt.savefig(outfile, dpi=300)
	plt.close()
	print(f"Saved plot to {outfile}")


if __name__ == "__main__":
	plot_i_frame_percentage_by_ti_group()
	plot_i_frame_percentage_by_ti_group_unweighted()



