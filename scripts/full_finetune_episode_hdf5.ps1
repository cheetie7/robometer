param(
    [string]$DatasetDir = ".",
    [string]$TaskDescription = "Move the two robotic arms together to insert the grasped red block into the blue block",
    [string]$ProcessedCacheDir = "processed_datasets",
    [string]$OutputDir = "logs",
    [string]$ExperimentName = "robometer4b_full_episode_hdf5",
    [int]$NumProcesses = 1,
    [int]$MaxSteps = 500,
    [int]$BatchSize = 1,
    [int]$GradAccum = 8,
    [string]$LogTo = "none"
)

$ErrorActionPreference = "Stop"

$DatasetKey = "datasets/episode_hdf5_rbm/episode_hdf5/episode_hdf5"
$env:ROBOMETER_PROCESSED_DATASETS_PATH = $ProcessedCacheDir

Write-Host "Step 1/3: converting episode_*.hdf5 files to a local Robometer dataset"
uv run python -m dataset_upload.generate_hf_dataset `
    --config_path dataset_upload/configs/data_gen_configs/episode_hdf5.yaml `
    --dataset.dataset_path="$DatasetDir" `
    --dataset.task_description="$TaskDescription" `
    --hub.push_to_hub=false

Write-Host "Step 2/3: preprocessing videos into Robometer cache"
uv run python -m robometer.data.scripts.preprocess_datasets `
    --config robometer/configs/preprocess_episode_hdf5.yaml `
    --cache_dir="$ProcessedCacheDir"

Write-Host "Step 3/3: full fine-tuning Robometer-4B"
uv run accelerate launch `
    --config_file robometer/configs/distributed/fsdp.yaml `
    --num_processes=$NumProcesses `
    train.py `
    model.base_model_id=Qwen/Qwen3-VL-4B-Instruct `
    model.use_peft=false `
    model.train_language_model=true `
    model.train_vision_encoder=false `
    model.train_progress_head=true `
    model.train_preference_head=true `
    model.train_success_head=true `
    data.train_datasets=[$DatasetKey] `
    data.eval_datasets=[$DatasetKey] `
    training.load_from_checkpoint=robometer/Robometer-4B `
    training.per_device_train_batch_size=$BatchSize `
    training.per_device_eval_batch_size=$BatchSize `
    training.gradient_accumulation_steps=$GradAccum `
    training.learning_rate=2e-5 `
    training.warmup_ratio=0.1 `
    training.weight_decay=0.01 `
    training.max_steps=$MaxSteps `
    training.output_dir="$OutputDir" `
    training.exp_name="$ExperimentName" `
    training.overwrite_output_dir=true `
    training.eval_steps=50 `
    training.custom_eval_steps=50 `
    logging.log_to=[$LogTo] `
    custom_eval.eval_types=[reward_alignment] `
    custom_eval.reward_alignment=[$DatasetKey] `
    logging.save_best.metric_names=[eval_rew_align/pearson_episode_hdf5] `
    logging.save_best.greater_is_better=[true]
