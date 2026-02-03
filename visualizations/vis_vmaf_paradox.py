"""Visualizations contrasting VMAF vs temporal stability (MOVIE Index) across TI groups."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["BVI-HD", "HEVC_CLASS_B", "UVG"]
CODECS = ["h264", "hevc", "vp9", "av1"]
LEVELS = [1, 1.5, 2, 2.5, 3, 4, 8, 12]

DATASET_PREFIX = {
	"UVG": "results/eval_metrics_uvg_",
	"HEVC_CLASS_B": "results/eval_metrics_hevc_class_b_",
	"BVI-HD": "results/eval_metrics_BVI-HD_",
}

TI_FILES = {
	"UVG": "results/eval_metrics_uvg_TI.json",
	"HEVC_CLASS_B": "results/eval_metrics_HEVC_CLASS_B_TI.json",
	"BVI-HD": "results/eval_metrics_BVI-HD_TI.json",
}


def _first_token(name: str) -> str:
	"""Use token before first underscore to align TI grouping and metric files."""
	return name.split("_")[0]


def load_ti_groups(
	datasets: Iterable[str], number_of_groups: int = 4
) -> Tuple[Dict[int, List[Tuple[str, float]]], List[float]]:
	entries: List[Tuple[str, float]] = []
	for dataset in datasets:
		ti_path = TI_FILES.get(dataset)
		if not ti_path or not os.path.exists(ti_path):
			print(f"Warning: TI file missing for dataset {dataset} -> {ti_path}")
			continue
		with open(ti_path, "r", encoding="utf-8") as f:
			ti_data = json.load(f)
		for video_name, ti_value in ti_data.items():
			entries.append((_first_token(video_name), float(ti_value)))

	if not entries:
		raise ValueError("No TI data found; cannot form groups.")

	sorted_entries = sorted(entries, key=lambda item: item[1])
	n = len(sorted_entries)
	bounds = [
		sorted_entries[min((i + 1) * n // number_of_groups - 1, n - 1)][1]
		for i in range(number_of_groups - 1)
	]

	groups: Dict[int, List[Tuple[str, float]]] = {
		i: [] for i in range(1, number_of_groups + 1)
	}
	for name, ti_value in entries:
		for idx, bound in enumerate(bounds):
			if ti_value <= bound:
				groups[idx + 1].append((name, ti_value))
				break
		else:
			groups[number_of_groups].append((name, ti_value))

	return groups, bounds


def collect_group_means(
	ti_groups: Dict[int, List[Tuple[str, float]]],
	datasets: Iterable[str],
	codecs: Iterable[str],
	levels: Iterable[float],
) -> Dict[int, Dict[float, Dict[str, float]]]:
	metrics: Dict[int, Dict[float, Dict[str, List[float]]]] = {}
	for group_id in ti_groups.keys():
		metrics[group_id] = {
			level: {"vmaf": [], "movie_index": []}
			for level in levels
		}

	for dataset in datasets:
		prefix = DATASET_PREFIX.get(dataset)
		if not prefix:
			print(f"Warning: dataset prefix missing for {dataset}")
			continue
		for codec in codecs:
			for level in levels:
				results_path = f"{prefix}{codec}_level{level}.json"
				if not os.path.exists(results_path):
					print(f"Skipping missing results file: {results_path}")
					continue
				with open(results_path, "r", encoding="utf-8") as f:
					video_results = json.load(f)

				for video_name, video_data in video_results.items():
					token = _first_token(video_name)
					for group_id, videos in ti_groups.items():
						group_tokens = {v[0] for v in videos}
						if token not in group_tokens:
							continue
						vmaf_val = video_data.get("vmaf")
						movie_val = video_data.get("movie_index")
						if vmaf_val is not None:
							metrics[group_id][level]["vmaf"].append(float(vmaf_val))
						if movie_val is not None:
							metrics[group_id][level]["movie_index"].append(float(movie_val))

	means: Dict[int, Dict[float, Dict[str, float]]] = {}
	for group_id, levels_dict in metrics.items():
		means[group_id] = {}
		for level, metric_lists in levels_dict.items():
			means[group_id][level] = {
				"vmaf": float(np.mean(metric_lists["vmaf"]))
				if metric_lists["vmaf"]
				else np.nan,
				"movie_index": float(np.mean(metric_lists["movie_index"]))
				if metric_lists["movie_index"]
				else np.nan,
			}
	return means


def collect_group_means_by_codec(
	ti_groups: Dict[int, List[Tuple[str, float]]],
	datasets: Iterable[str],
	codecs: Iterable[str],
	levels: Iterable[float],
) -> Dict[str, Dict[int, Dict[float, Dict[str, float]]]]:
	metrics: Dict[str, Dict[int, Dict[float, Dict[str, List[float]]]]] = {}
	for codec in codecs:
		metrics[codec] = {
			group_id: {
				level: {"vmaf": [], "movie_index": []}
				for level in levels
			}
			for group_id in ti_groups.keys()
		}

	for dataset in datasets:
		prefix = DATASET_PREFIX.get(dataset)
		if not prefix:
			print(f"Warning: dataset prefix missing for {dataset}")
			continue
		for codec in codecs:
			for level in levels:
				results_path = f"{prefix}{codec}_level{level}.json"
				if not os.path.exists(results_path):
					print(f"Skipping missing results file: {results_path}")
					continue
				with open(results_path, "r", encoding="utf-8") as f:
					video_results = json.load(f)

				for video_name, video_data in video_results.items():
					token = _first_token(video_name)
					for group_id, videos in ti_groups.items():
						group_tokens = {v[0] for v in videos}
						if token not in group_tokens:
							continue
						vmaf_val = video_data.get("vmaf")
						movie_val = video_data.get("movie_index")
						if vmaf_val is not None:
							metrics[codec][group_id][level]["vmaf"].append(float(vmaf_val))
						if movie_val is not None:
							metrics[codec][group_id][level]["movie_index"].append(float(movie_val))

	means: Dict[str, Dict[int, Dict[float, Dict[str, float]]]] = {}
	for codec, groups_dict in metrics.items():
		means[codec] = {}
		for group_id, levels_dict in groups_dict.items():
			means[codec][group_id] = {}
			for level, metric_lists in levels_dict.items():
				means[codec][group_id][level] = {
					"vmaf": float(np.mean(metric_lists["vmaf"]))
					if metric_lists["vmaf"]
					else np.nan,
					"movie_index": float(np.mean(metric_lists["movie_index"]))
					if metric_lists["movie_index"]
					else np.nan,
				}
	return means


def _level_to_bitrate(level: float) -> int:
	return int(level * 1000)


def plot_percent_drop(
	means: Dict[int, Dict[float, Dict[str, float]]],
	levels: Iterable[float],
	output_path: str,
) -> None:
	fig, ax = plt.subplots(figsize=(9, 5))
	colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

	for group_id, color in colors.items():
		if group_id not in means:
			continue
		movie_vals = [means[group_id].get(level, {}).get("movie_index", np.nan) for level in levels]
		base = movie_vals[0] if movie_vals else np.nan
		if np.isnan(base):
			print(f"Skipping TI Group {group_id}: missing baseline MOVIE Index")
			continue
		pct_drop = [100 * (base - v) / base if not np.isnan(v) else np.nan for v in movie_vals]
		bitrates = [_level_to_bitrate(lvl) for lvl in levels]
		ax.plot(bitrates, pct_drop, marker="o", color=color, label=f"TI Group {group_id}")

	ax.set_xlabel("Bitrate (kbps)")
	ax.set_ylabel("% Decrease in MOVIE Index vs. lowest bitrate")
	ax.set_title("Relative MOVIE Index drop by bitrate")
	ax.grid(True, linestyle="--", alpha=0.35)
	ax.legend()
	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"Saved percent-drop plot to {output_path}")


def plot_efficiency_loss(
	means_by_codec: Dict[str, Dict[int, Dict[float, Dict[str, float]]]],
	low_bitrate: int,
	high_bitrate: int,
	output_path: str,
) -> None:
	level_by_bitrate = {_level_to_bitrate(lvl): lvl for lvl in LEVELS}
	low_level = level_by_bitrate.get(low_bitrate)
	high_level = level_by_bitrate.get(high_bitrate)
	if low_level is None or high_level is None:
		raise ValueError("Requested bitrate not in LEVELS mapping")

	codecs = list(means_by_codec.keys())
	groups = sorted({gid for codec_means in means_by_codec.values() for gid in codec_means.keys()})
	x = np.arange(len(groups))
	bar_width = 0.18

	fig, ax = plt.subplots(figsize=(9, 5))

	for idx, codec in enumerate(codecs):
		offset = (idx - (len(codecs) - 1) / 2) * bar_width
		heights = []
		label_groups = []
		for group_id in groups:
			group_means = means_by_codec.get(codec, {}).get(group_id, {})
			low_val = group_means.get(low_level, {}).get("movie_index", np.nan) if low_level else np.nan
			high_val = group_means.get(high_level, {}).get("movie_index", np.nan) if high_level else np.nan
			if np.isnan(low_val) or np.isnan(high_val):
				heights.append(np.nan)
				label_groups.append(group_id)
				continue
            # relative delta calculation
			delta = (low_val - high_val)/low_val * 100
			heights.append(delta)
			label_groups.append(group_id)
		positions = x + offset
		ax.bar(positions, heights, bar_width, label=codec.upper())

	ax.set_xticks(x)
	ax.set_xticklabels([f"TI Group {g}" for g in groups])
	ax.set_ylabel("Efficiency Loss in MOVIE Index (%)")
	#ax.set_title(f"Efficiency Loss in MOVIE Index: {low_bitrate} kbps → {high_bitrate} kbps")
	ax.grid(axis="y", linestyle="--", alpha=0.35)
	ax.legend()
	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"Saved efficiency-loss plot to {output_path}")


def plot_convergence(
	means: Dict[int, Dict[float, Dict[str, float]]],
	groups: Tuple[int, int],
	levels: Iterable[float],
	output_path: str,
) -> None:
	fig, ax_left = plt.subplots(figsize=(9, 5))
	ax_right = ax_left.twinx()

	colors = {groups[0]: "#2ca02c", groups[1]: "#d62728"}

	bitrates = [_level_to_bitrate(lvl) for lvl in levels]

	for group_id in groups:
		vmaf_vals = [means[group_id].get(level, {}).get("vmaf", np.nan) for level in levels]
		movie_vals = [means[group_id].get(level, {}).get("movie_index", np.nan) for level in levels]
		ax_left.plot(
			bitrates,
			vmaf_vals,
			marker="o",
			color=colors[group_id],
			linestyle="-",
			label=f"VMAF G{group_id}",
		)
		ax_right.plot(
			bitrates,
			movie_vals,
			marker="s",
			color=colors[group_id],
			linestyle="--",
			label=f"MOVIE G{group_id}",
		)

	ax_left.set_xlabel("Bitrate (kbps)")
	ax_left.set_ylabel("VMAF", color="#1f1f1f")
	ax_right.set_ylabel("MOVIE Index", color="#1f1f1f")
	ax_left.set_title("VMAF vs MOVIE convergence (TI Groups 3 vs 4)")
	ax_left.grid(True, linestyle="--", alpha=0.35)

	handles_left, labels_left = ax_left.get_legend_handles_labels()
	handles_right, labels_right = ax_right.get_legend_handles_labels()
	ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper center")

	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"Saved convergence plot to {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate VMAF/MOVIE visuals highlighting temporal vs spatial behavior."
	)
	parser.add_argument(
		"--datasets",
		default=",".join(DATASETS),
		help="Comma-separated datasets to include (default: all).",
	)
	parser.add_argument(
		"--codecs",
		default=",".join(CODECS),
		help="Comma-separated codecs to include (default: all).",
	)
	parser.add_argument(
		"--output-dir",
		default="visualizations",
		help="Directory to place generated figures.",
	)
	parser.add_argument(
		"--low-bitrate",
		type=int,
		default=2000,
		help="Low bitrate (kbps) for efficiency-loss bar chart (must map to LEVELS).",
	)
	parser.add_argument(
		"--high-bitrate",
		type=int,
		default=8000,
		help="High bitrate (kbps) for efficiency-loss bar chart (must map to LEVELS).",
	)
	args = parser.parse_args()

	selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
	selected_codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

	ti_groups, bounds = load_ti_groups(selected_datasets)
	print(f"TI bounds: {bounds}")

	means = collect_group_means(
		ti_groups=ti_groups,
		datasets=selected_datasets,
		codecs=selected_codecs,
		levels=LEVELS,
	)
	means_by_codec = collect_group_means_by_codec(
		ti_groups=ti_groups,
		datasets=selected_datasets,
		codecs=selected_codecs,
		levels=LEVELS,
	)

	os.makedirs(args.output_dir, exist_ok=True)

	plot_percent_drop(
		means=means,
		levels=LEVELS,
		output_path=os.path.join(args.output_dir, "vmaf_paradox_percent_drop.png"),
	)

	plot_efficiency_loss(
		means_by_codec=means_by_codec,
		low_bitrate=args.low_bitrate,
		high_bitrate=args.high_bitrate,
		output_path=os.path.join(args.output_dir, "vmaf_paradox_efficiency_loss.png"),
	)

	plot_convergence(
		means=means,
		groups=(3, 4),
		levels=LEVELS,
		output_path=os.path.join(args.output_dir, "vmaf_paradox_convergence.png"),
	)


if __name__ == "__main__":
	main()

