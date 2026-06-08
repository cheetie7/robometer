#!/usr/bin/env python3
"""Loader for episode_*.hdf5 trajectories used in local Robometer fine-tuning."""

from pathlib import Path
import uuid

import h5py
import numpy as np
from tqdm import tqdm


FRAME_DATASET = "observations/images/cam_high"


class EpisodeHDF5FrameLoader:
    """Pickle-able lazy frame loader for one HDF5 trajectory."""

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path

    def __call__(self) -> np.ndarray:
        with h5py.File(self.hdf5_path, "r") as f:
            return f[FRAME_DATASET][:]


def load_episode_hdf5_dataset(
    dataset_path: str,
    dataset_name: str = "episode_hdf5",
    task_description: str = "Move the two robotic arms together to insert the grasped red block into the blue block",
    max_trajectories: int | None = None,
) -> dict[str, list[dict]]:
    """Load one trajectory per episode_*.hdf5 file."""

    root = Path(dataset_path).expanduser()
    hdf5_files = sorted(root.glob("episode_*.hdf5"))

    if max_trajectories not in (None, -1):
        hdf5_files = hdf5_files[:max_trajectories]

    print(f"Loading Episode HDF5 dataset from: {root}")
    print(f"Found {len(hdf5_files)} HDF5 trajectory files")

    trajectories = []
    for file_path in tqdm(hdf5_files, desc=f"Processing {dataset_name}"):
        trajectories.append(
            {
                "id": str(uuid.uuid4()),
                "frames": EpisodeHDF5FrameLoader(str(file_path)),
                "actions": np.empty((0,), dtype=np.float32),
                "is_robot": True,
                "task": task_description,
                "quality_label": "successful",
                "data_source": dataset_name,
                "partial_success": None,
                "preference_group_id": None,
                "preference_rank": None,
            }
        )

    print(f"Loaded {len(trajectories)} trajectories from 1 task")
    return {task_description: trajectories}
