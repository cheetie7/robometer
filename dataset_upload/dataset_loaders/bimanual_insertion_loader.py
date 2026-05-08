#!/usr/bin/env python3
"""Loader for local bimanual shaft-hole insertion HDF5 episodes."""

from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from tqdm import tqdm


TASK_DESCRIPTION = "Use both robot arms to insert the shaft into the hole."
DATA_SOURCE = "bimanual_insertion"


class BimanualInsertionFrameLoader:
    """Pickle-able loader that reads top-view RGB frames from one HDF5 episode."""

    def __init__(self, hdf5_path: str, dataset_path: str = "observations/images/top"):
        self.hdf5_path = hdf5_path
        self.dataset_path = dataset_path

    def __call__(self) -> np.ndarray:
        with h5py.File(self.hdf5_path, "r") as f:
            if self.dataset_path not in f:
                raise KeyError(f"Dataset path '{self.dataset_path}' not found in {self.hdf5_path}")
            frames = f[self.dataset_path][:]

        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frame shape in {self.hdf5_path}: {getattr(frames, 'shape', None)}")

        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        return frames


def _load_actions(file_path: Path) -> np.ndarray | None:
    with h5py.File(file_path, "r") as f:
        if "action" in f:
            return f["action"][:]
        if "actions" in f:
            return f["actions"][:]
    return None


def load_bimanual_insertion_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
) -> dict[str, list[dict]]:
    """Load local HDF5 episodes for Robometer dataset conversion."""

    root = Path(dataset_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Bimanual insertion dataset path not found: {root}")

    hdf5_files = sorted(root.glob("*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(root.rglob("*.hdf5"))
    if max_trajectories is not None and max_trajectories != -1:
        hdf5_files = hdf5_files[:max_trajectories]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 episodes found under {root}")

    trajectories: list[dict] = []
    for file_path in tqdm(hdf5_files, desc="Loading bimanual insertion episodes"):
        trajectory = {
            "frames": BimanualInsertionFrameLoader(str(file_path)),
            "actions": _load_actions(file_path),
            "is_robot": True,
            "task": TASK_DESCRIPTION,
            "quality_label": "successful",
            "data_source": DATA_SOURCE,
            "partial_success": 1.0,
            "id": str(uuid4()),
        }
        trajectories.append(trajectory)

    return {DATA_SOURCE: trajectories}
