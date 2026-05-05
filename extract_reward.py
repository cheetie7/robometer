from __future__ import annotations
import subprocess
from pathlib import Path


import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

def batch_process_videos(video_dir: str, model_path: str, default_task: str):
    dir_path=Path(video_dir)
    video_files=list(dir_path.rglob("*.mp4"))
    for index,video in enumerate(video_files,start=1):
        cmd = [
            "uv", "run", "python", "scripts/example_inference_local.py",
            "--model-path", model_path,
            "--video", str(video),
            "--task", default_task
        ]
        try:
            # 执行命令，check=True 表示如果命令报错会抛出异常
            subprocess.run(cmd, check=True)
            print(f"✅ 处理完成: {video.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ 处理失败: {video.name}")
            print(f"错误详情: {e}\n")

def load_frames_input(
    video_or_array_path: str,
    *,
    fps: float = 1.0,
    max_frames: int = 20,
) -> np.ndarray:
    """Load frames from a video path/URL or .npy/.npz file. Returns uint8 (T, H, W, C)."""
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "frames" in npz:
                frames_array = npz["frames"].copy()
            elif "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
    else:
        frames_array = extract_frames(video_or_array_path, fps=fps, max_frames=max_frames)
        if frames_array is None or frames_array.size == 0:
            raise RuntimeError("Could not extract frames from video.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4 and frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
        frames_array = frames_array.transpose(0, 2, 3, 1)
    return frames_array


def compute_rewards_per_frame_local(
    reward_model,
    processor, 
    tokenizer, 
    exp_config,
    video_frames: np.ndarray,
    task: str,
    device: Optional[torch.device] = None,
    

) -> Tuple[np.ndarray, np.ndarray]:
    """Load RBM from HuggingFace and run inference; return per-frame progress and success arrays."""
    
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    T = int(video_frames.shape[0])
    traj = Trajectory(
        frames=video_frames,
        frames_shape=tuple(video_frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    results = compute_batch_outputs(
        reward_model,
        tokenizer,
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=is_discrete,
        num_bins=num_bins,
    )

    progress_pred = results.get("progress_pred", [])
    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.array([], dtype=np.float32)
    )

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.array([], dtype=np.float32)
    )

    return progress_array, success_array


def set_config_max_frames(exp_config, max_frames: int) -> None:
    """Keep the loaded model experiment config aligned with inference frame sampling."""
    if not hasattr(exp_config, "data") or exp_config.data is None:
        raise ValueError("Loaded model config does not contain a data section.")
    exp_config.data.max_frames = int(max_frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RBM inference locally: load model from HuggingFace and compute per-frame progress and success.",
        epilog="Outputs: <out>.npy (rewards), <out>_success_probs.npy, <out>_progress_success.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model id or local checkpoint path")
    
    parser.add_argument("--task", required=True, help="Task instruction for the trajectory")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS when sampling from video (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=20, help="Max frames to extract from video (default: 20)")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success in plot (default: 0.5)",
    )
    parser.add_argument("--out", default=None, help="Output path for rewards .npy (default: <video_stem>_rewards.npy)")
    parser.add_argument("--video_dir",required=True)
    args = parser.parse_args()

    
    video_path=Path(args.video_dir)
    
    video_files=list(video_path.rglob("*.mp4"))
    device=None
    model_path=args.model_path
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    set_config_max_frames(exp_config, int(args.max_frames))
    reward_model.eval()
    for idx,video in enumerate(video_files,start=1):
        out_path = Path(args.out) if args.out is not None else video.with_name(video.stem + "_rewards.npy")
        frames = load_frames_input(
        str(video),
        fps=float(args.fps),
        max_frames=int(args.max_frames),
        )

        rewards, success_probs = compute_rewards_per_frame_local(
            reward_model=reward_model,
            processor=processor,
            tokenizer=tokenizer,
            exp_config=exp_config,
            video_frames=frames,
            task=args.task,
            device='cuda'
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), rewards)
        success_path = out_path.with_name(out_path.stem + "_success_probs.npy")
        np.save(str(success_path), success_probs)

        show_success = success_probs.size > 0 and success_probs.size == rewards.size
        success_binary = (success_probs > float(args.success_threshold)).astype(np.int32) if show_success else None
        fig = create_combined_progress_success_plot(
            progress_pred=rewards,
            num_frames=int(frames.shape[0]),
            success_binary=success_binary,
            success_probs=success_probs if show_success else None,
            success_labels=None,
            title=f"Progress/Success — {video_path.name}",
        )
        plot_path = out_path.with_name(out_path.stem + "_progress_success.png")
        fig.savefig(str(plot_path), dpi=200)
        plt.close(fig)

        summary = {
            "video": str(video_path),
            "num_frames": int(frames.shape[0]),
            "model_path": args.model_path,
            "out_rewards": str(out_path),
            "out_success_probs": str(success_path),
            "out_plot": str(plot_path),
            "reward_min": float(np.min(rewards)) if rewards.size else None,
            "reward_max": float(np.max(rewards)) if rewards.size else None,
            "reward_mean": float(np.mean(rewards)) if rewards.size else None,
        }
        print(json.dumps(summary, indent=2))




if __name__ == "__main__":
    main()
