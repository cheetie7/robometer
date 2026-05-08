#!/usr/bin/env python3
"""Quick smoke test for the bimanual insertion dataset loader."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_upload.dataset_loaders.bimanual_insertion_loader import load_bimanual_insertion_dataset


def flatten_task_data(task_data: dict[str, list[dict]]) -> list[dict]:
    trajectories = []
    for task_name, task_trajectories in task_data.items():
        for trajectory in task_trajectories:
            trajectory["task_name"] = task_name
            trajectories.append(trajectory)
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data", help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--max-trajectories", type=int, default=1)
    args = parser.parse_args()

    task_data = load_bimanual_insertion_dataset(args.dataset_path, max_trajectories=args.max_trajectories)
    trajectories = flatten_task_data(task_data)
    if not trajectories:
        raise RuntimeError("No trajectories loaded")

    sample = trajectories[0]
    frames = sample["frames"]()
    actions = sample["actions"]

    print(f"Loaded trajectories: {len(trajectories)}")
    print(f"Sample keys: {sorted(sample.keys())}")
    print(f"Sample task: {sample['task']}")
    print(f"Frames shape: {frames.shape}, dtype={frames.dtype}")
    print(f"Actions shape: {getattr(actions, 'shape', None)}")
    print(f"Quality label: {sample['quality_label']}")
    print(f"Data source: {sample['data_source']}")


if __name__ == "__main__":
    main()
