#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-.}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Move the two robotic arms together to insert the grasped red block into the blue block}"
PROCESSED_CACHE_DIR="${PROCESSED_CACHE_DIR:-processed_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-logs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-robometer4b_full_episode_hdf5}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_STEPS="${MAX_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOG_TO="${LOG_TO:-none}"
DATASET_KEY="datasets/episode_hdf5_rbm/episode_hdf5/episode_hdf5"

export ROBOMETER_PROCESSED_DATASETS_PATH="$PROCESSED_CACHE_DIR"

echo "Step 1/3: converting episode_*.hdf5 files to a local Robometer dataset"
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path dataset_upload/configs/data_gen_configs/episode_hdf5.yaml \
  --dataset.dataset_path="$DATASET_DIR" \
  --dataset.task_description="$TASK_DESCRIPTION" \
  --hub.push_to_hub=false

echo "Step 2/3: preprocessing videos into Robometer cache"
uv run python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess_episode_hdf5.yaml \
  --cache_dir="$PROCESSED_CACHE_DIR"

echo "Step 3/3: full fine-tuning Robometer-4B"
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes="$NUM_PROCESSES" \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_language_model=true \
  model.train_vision_encoder=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=true \
  data.train_datasets=[$DATASET_KEY] \
  data.eval_datasets=[$DATASET_KEY] \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size="$BATCH_SIZE" \
  training.per_device_eval_batch_size="$BATCH_SIZE" \
  training.gradient_accumulation_steps="$GRAD_ACCUM" \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps="$MAX_STEPS" \
  training.output_dir="$OUTPUT_DIR" \
  training.exp_name="$EXPERIMENT_NAME" \
  training.overwrite_output_dir=true \
  training.eval_steps=50 \
  training.custom_eval_steps=50 \
  logging.log_to=[$LOG_TO] \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=[$DATASET_KEY] \
  logging.save_best.metric_names=[eval_rew_align/pearson_episode_hdf5] \
  logging.save_best.greater_is_better=[true]
