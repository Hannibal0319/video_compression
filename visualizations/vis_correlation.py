"""Scatter plot of VMAF vs a temporal metric (MOVIE Index or ST-RRED), colored by TI group."""

import argparse
import json
import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from vis_vmaf_paradox import (
	CODECS,
	DATASETS,
	DATASET_PREFIX,
	LEVELS,
	_first_token,
	load_ti_groups,
)


def _ti_lookup(ti_groups: Dict[int, List[tuple]]) -> Dict[str, int]:
	return {name: group_id for group_id, pairs in ti_groups.items() for name, _ in pairs}


def collect_points(
	ti_lookup: Dict[str, int],
	datasets: Iterable[str],
	codecs: Iterable[str],
	levels: Iterable[float],
	y_metric: str,
) -> List[Dict[str, object]]:
	points: List[Dict[str, object]] = []
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
					group_id = ti_lookup.get(token)
					if group_id is None:
						continue
					vmaf = video_data.get("vmaf")
					y_val = video_data.get(y_metric)
					if vmaf is None or y_val is None:
						continue
					points.append(
						{
							"vmaf": float(vmaf),
							"y": float(y_val),
							"ti_group": group_id,
							"dataset": dataset,
							"codec": codec,
							"level": level,
							"video": token,
						}
					)
	return points


def plot_vmaf_vs_temporal(
	points: List[Dict[str, object]],
	output_path: str,
	y_label: str,
	invert_y_axis: bool,
	title: str,
) -> None:
	fig, ax = plt.subplots(figsize=(9, 5.5))
	colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

	for group_id in sorted({p["ti_group"] for p in points}):
		group_points = [p for p in points if p["ti_group"] == group_id]
		x_vals = [p["vmaf"] for p in group_points]
		y_vals = [p["y"] for p in group_points]
		ax.scatter(
			x_vals,
			y_vals,
			label=f"TI Group {group_id}",
			color=colors.get(group_id, "gray"),
			alpha=0.75,
			edgecolors="white",
			linewidth=0.6,
		)

	ax.set_xlabel("VMAF")
	ax.set_ylabel(y_label)
	ax.set_title(title)
	ax.grid(True, linestyle="--", alpha=0.35)
	if invert_y_axis:
		ax.invert_yaxis()

	legend = ax.legend(title="TI Groups")
	legend.get_frame().set_alpha(0.9)

	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"Saved scatter plot to {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot VMAF vs a temporal metric (movie_index or st_rred), color-coded by TI group."
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
		"--output-name",
		default=None,
		help="Filename for the scatter plot (defaults to vmaf_vs_<metric>_by_ti.png).",
	)
	parser.add_argument(
		"--invert-y-axis",
		action="store_true",
		help="Invert temporal metric axis so better scores appear higher.",
	)
	parser.add_argument(
		"--y-metric",
		choices=["movie_index", "st_rred"],
		default="movie_index",
		help="Temporal metric to plot against VMAF.",
	)
	args = parser.parse_args()

	selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
	selected_codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
	y_metric = args.y_metric
	y_label = "MOVIE Index" if y_metric == "movie_index" else "ST-RRED"
	title = "VMAF vs. MOVIE Index by TI Group" if y_metric == "movie_index" else "VMAF vs. ST-RRED by TI Group"

	ti_groups, bounds = load_ti_groups(selected_datasets)
	print(f"TI bounds: {bounds}")
	ti_lookup = _ti_lookup(ti_groups)
	points = collect_points(
		ti_lookup=ti_lookup,
		datasets=selected_datasets,
		codecs=selected_codecs,
		levels=LEVELS,
		y_metric=y_metric,
	)

	if not points:
		raise ValueError("No data points found to plot.")

	os.makedirs(args.output_dir, exist_ok=True)
	output_name = args.output_name or f"vmaf_vs_{y_metric}_by_ti.png"
	output_path = os.path.join(args.output_dir, output_name)
	plot_vmaf_vs_temporal(
		points=points,
		output_path=output_path,
		y_label=y_label,
		invert_y_axis=args.invert_y_axis,
		title=None,
	)


if __name__ == "__main__":
	main()
