from __future__ import annotations
import subprocess
from pathlib import Path


import argparse
import decord
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def extract_uniform_frames(video_path: str, num_frames: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract exactly num_frames uniformly across a video and return sampled seconds/indices."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"Could not read frames from video: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, int(num_frames), dtype=int).tolist()
    frames_array = vr.get_batch(frame_indices).asnumpy()
    try:
        native_fps = float(vr.get_avg_fps())
    except Exception:
        native_fps = 0.0
    if native_fps > 0:
        frame_times = np.asarray(frame_indices, dtype=np.float32) / native_fps
    else:
        frame_times = np.arange(len(frame_indices), dtype=np.float32)
    del vr
    return frames_array, frame_times, frame_indices


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
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load frames and x-axis times. Returns uint8 (T, H, W, C), seconds, and source frame indices."""
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
        frame_times = np.arange(frames_array.shape[0], dtype=np.float32)
        frame_indices = list(range(frames_array.shape[0]))
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "frames" in npz:
                frames_array = npz["frames"].copy()
            elif "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
        frame_times = np.arange(frames_array.shape[0], dtype=np.float32)
        frame_indices = list(range(frames_array.shape[0]))
    else:
        frames_array, frame_times, frame_indices = extract_uniform_frames(video_or_array_path, num_frames=max_frames)
        if frames_array is None or frames_array.size == 0:
            raise RuntimeError("Could not extract frames from video.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4 and frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
        frames_array = frames_array.transpose(0, 2, 3, 1)
    return frames_array, frame_times, frame_indices


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


def configure_inference_frames(exp_config, reward_model, max_frames: int) -> None:
    """Keep inference config aligned with frame sampling and request one prediction per frame."""
    if not hasattr(exp_config, "data") or exp_config.data is None:
        raise ValueError("Loaded model config does not contain a data section.")
    exp_config.data.max_frames = int(max_frames)
    exp_config.data.use_multi_image = True
    exp_config.model.use_multi_image = True
    reward_model.use_multi_image = True
    reward_model.model_config.use_multi_image = True


def align_times_to_predictions(frame_times: np.ndarray, num_predictions: int) -> np.ndarray:
    """Return x-axis seconds aligned to prediction count while covering the sampled video span."""
    if num_predictions <= 0:
        return np.array([], dtype=np.float32)
    if len(frame_times) == num_predictions:
        return frame_times
    if len(frame_times) == 0:
        return np.arange(num_predictions, dtype=np.float32)
    if len(frame_times) == 1:
        return np.full(num_predictions, float(frame_times[0]), dtype=np.float32)
    return np.linspace(float(frame_times[0]), float(frame_times[-1]), num_predictions, dtype=np.float32)


def set_plot_x_axis_to_seconds(fig, x_values: np.ndarray) -> None:
    """Replace frame-index x data in the progress/success plot with video seconds."""
    if len(x_values) == 0:
        return
    formatter = FuncFormatter(lambda value, _: f"{value:.1f}s")
    for ax in fig.axes:
        for line in ax.lines:
            if len(line.get_xdata()) == len(x_values):
                line.set_xdata(x_values)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_xlabel("Video time (s)")
        ax.xaxis.set_major_formatter(formatter)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RBM inference locally: load model from HuggingFace and compute per-frame progress and success.",
        epilog="Outputs: <out>.npy (rewards), <out>_success_probs.npy, <out>_progress_success.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model id or local checkpoint path")
    
    parser.add_argument("--task", required=True, help="Task instruction for the trajectory")
    parser.add_argument("--fps", type=float, default=1.0, help="Unused for video input; frames are sampled uniformly")
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
    print(f"[extract_reward] effective max_frames={int(args.max_frames)}")

    
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
    configure_inference_frames(exp_config, reward_model, int(args.max_frames))
    reward_model.eval()
    for idx,video in enumerate(video_files,start=1):
        out_path = Path(args.out) if args.out is not None else video.with_name(video.stem + "_rewards.npy")
        frames, frame_times, frame_indices = load_frames_input(
        str(video),
        fps=float(args.fps),
        max_frames=int(args.max_frames),
        )
        print(
            f"[extract_reward] video={video.name} sampled_input_frames={int(frames.shape[0])} "
            f"sampled_frame_indices={frame_indices}"
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
        plot_times = align_times_to_predictions(frame_times, rewards.size)
        print(
            f"[extract_reward] video={video.name} reward_count={int(rewards.size)} "
            f"success_count={int(success_probs.size)}"
        )
        fig = create_combined_progress_success_plot(
            progress_pred=rewards,
            num_frames=int(frames.shape[0]),
            success_binary=success_binary,
            success_probs=success_probs if show_success else None,
            success_labels=None,
            title=f"Progress/Success — {video_path.name}",
        )
        set_plot_x_axis_to_seconds(fig, plot_times)
        plot_path = out_path.with_name(out_path.stem + "_progress_success.png")
        fig.savefig(str(plot_path), dpi=200)
        plt.close(fig)

        summary = {
            "video": str(video_path),
            "num_frames": int(frames.shape[0]),
            "reward_count": int(rewards.size),
            "sampled_frame_indices": frame_indices,
            "sampled_times_seconds": frame_times.tolist(),
            "plot_times_seconds": plot_times.tolist(),
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
