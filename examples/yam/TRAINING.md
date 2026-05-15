# YAM Pi0.5 LoRA Training

Fine-tunes the pi0.5 base model on YAM teleop data using LoRA.
Requires a converted LeRobot dataset (see [DATA_CONVERSION.md](DATA_CONVERSION.md)).

## Prerequisites

- Converted LeRobot dataset at `~/.cache/huggingface/lerobot/local/yam_simpletest/`
- GPU with sufficient VRAM (LoRA reduces memory vs full fine-tuning)
- `conda deactivate` before running any `uv` commands

## Step 1: Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_yam_simpletest_lora
```

Stats are saved to `assets/pi05_yam_simpletest_lora/local/yam_simpletest/`.

### What happens in this step

The model needs normalized inputs. This script computes the statistics used for normalization
by iterating through the entire dataset with the same transform pipeline used during training:

1. **Repack transforms** run first — mapping LeRobot field names (`observation.state`, `action`)
   to openpi's internal names (`state`, `actions`). Note: the LeRobot dataset stores the key
   as `action` (singular), which is configured via `action_sequence_keys=("action",)` in the
   training config.
2. **YamInputs** runs next — mapping camera names to the model's expected keys (`top` ->
   `base_0_rgb`, etc.) and setting image masks. This is a custom transform we wrote
   (`src/openpi/policies/yam_policy.py`) instead of using the built-in `AlohaInputs`, which
   hardcodes Aloha-specific camera names and joint/gripper transformations.
3. **DeltaActions** transform subtracts the current state from each action's joint dimensions
   (grippers stay absolute). This means the norm stats are computed on **delta actions**, not
   the raw absolute actions stored in the dataset.
4. **Running statistics** are accumulated over all frames, computing:
   - Mean and standard deviation (used by pi0 z-score normalization)
   - 1st and 99th percentile quantiles (used by pi0.5 quantile normalization)

Pi0.5 uses **quantile normalization** (not z-score). At training time, state and action values
are scaled based on the q01/q99 range, which is more robust to outliers than mean/std. The
`use_quantile_norm` flag is set automatically for pi0.5 based on the model type.

## Step 2: Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam_simpletest_lora --exp-name=yam_lora_v1 --wandb-enabled --overwrite
```

JAX auto-detects multiple GPUs and uses data parallelism.

Useful flags:
- `--wandb-enabled` — enable Weights & Biases logging
- `--overwrite` — overwrite existing checkpoint directory (useful when re-running training)
- `--resume` — resume training from the last checkpoint instead of starting fresh

Checkpoints are saved to `checkpoints/pi05_yam_simpletest_lora/yam_lora_v1/`.

### What happens in this step

Training loads the pi0.5 base model from Google Cloud Storage and fine-tunes it with LoRA
(Low-Rank Adaptation) on your dataset:

1. **Base model weights are downloaded** from `gs://openpi-assets/checkpoints/pi05_base/params`
   and cached locally in `~/.cache/openpi/`. The pi0.5 model consists of:
   - **PaliGemma** (2B params): A vision-language model (SigLIP vision encoder + Gemma 2B LLM)
     that processes camera images and the text prompt.
   - **Action expert** (300M params): A smaller Gemma LLM that generates action predictions
     via flow matching (iterative denoising from noise to action).

2. **LoRA adapters are injected** into both the PaliGemma LLM (rank 16) and the action expert
   (rank 32). LoRA adds small trainable low-rank matrices (w_a, w_b) to the attention and
   feed-forward layers. All original LLM weights are frozen — only the LoRA parameters are
   trained. This drastically reduces memory usage and training time compared to full fine-tuning.

3. **For each training step**, the data pipeline processes a batch:
   - LeRobot dataset returns a frame with the next 60 actions (action horizon = 60 at 60fps =
     1.0 second of future actions). This is called **action chunking** — the model predicts
     a sequence of future actions, not just the next one.
   - **Repack**: Maps dataset keys to internal format.
   - **YamInputs**: Maps camera names to model keys (`top` -> `base_0_rgb`, `left_wrist` ->
     `left_wrist_0_rgb`, `right_wrist` -> `right_wrist_0_rgb`) and sets image validity masks.
   - **DeltaActions**: Subtracts current state from joint action targets. The model learns to
     predict "how far to move" rather than absolute positions. Gripper values stay absolute
     (open/close is naturally absolute). The same current state is subtracted from all 60
     actions in the chunk.
   - **Quantile normalization**: Scales state and delta-actions to [-1, 1] using precomputed
     q01/q99 range.
   - **Image preprocessing**: Images are already 224x224. During training, augmentation is
     applied — random crop to 95% then resize back (non-wrist cameras only), random rotation
     +/-5 degrees (non-wrist cameras only), and color jitter (all cameras). The model
     distinguishes wrist vs non-wrist cameras by checking if the key name contains "wrist" —
     our naming (`left_wrist_0_rgb`, `right_wrist_0_rgb`) matches this convention correctly.
   - **Prompt tokenization**: The task string is tokenized by the PaliGemma tokenizer.

4. **Flow matching loss**: The model is trained with a flow matching objective. Random noise
   is added to the ground-truth action chunk, and the model predicts the velocity field to
   denoise it. The loss is MSE between predicted and true velocity.

5. **At inference time**, the reverse happens: the model starts from pure noise and iteratively
   denoises to produce an action chunk. Delta actions are converted back to absolute by adding
   the current state (`AbsoluteActions` transform), `YamOutputs` truncates to 14 dims, and
   quantile normalization is inverted.

## Config Details

The training config `pi05_yam_simpletest_lora` is defined in `src/openpi/training/config.py`.
It uses `SimpleDataConfig` with custom YAM transforms instead of the built-in
`LeRobotAlohaDataConfig`.

| Parameter | Value |
|-----------|-------|
| Base model | pi0.5 (`pi05=True`) |
| Config type | `SimpleDataConfig` with custom `YamInputs`/`YamOutputs` |
| LoRA rank | 16 (PaliGemma), 32 (action expert) |
| Frozen layers | All LLM weights except LoRA parameters |
| EMA | Disabled (standard for LoRA) |
| Normalization | Quantile (q01/q99), not z-score |
| Delta actions | Enabled for joints, grippers stay absolute |
| FPS | 60 |
| Action horizon | 60 (1.0 second at 60fps) |
| Training steps | 10,000 |
| Batch size | 32 |

### Transform Pipeline (training)

1. **Repack**: Maps LeRobot keys to internal format (`observation.images.top` -> `images["top"]`, etc.)
2. **YamInputs** (`src/openpi/policies/yam_policy.py`): Maps camera names to model keys (`top` -> `base_0_rgb`, `left_wrist` -> `left_wrist_0_rgb`, `right_wrist` -> `right_wrist_0_rgb`) and sets image masks.
3. **DeltaActions**: Converts joint actions to deltas relative to current state. Mask: `(T,T,T,T,T,T,F,T,T,T,T,T,T,F)` — 6 left joints (delta), left gripper (absolute), 6 right joints (delta), right gripper (absolute).
4. **Normalize**: Applies computed norm stats (quantile normalization on delta actions)
5. **ResizeImages(224, 224)**: No-op since images are already 224x224 from conversion
6. **TokenizePrompt**: Tokenizes task description
7. **Model augmentation** (training only): Random crop 95%, rotation, color jitter

### Why not use the built-in Aloha config?

openpi's `LeRobotAlohaDataConfig` uses `AlohaInputs` which:
- Hardcodes Aloha camera names (`cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`)
  and rejects any other names
- Maps cameras to model keys using Aloha conventions (`cam_high` -> `base_0_rgb`)
- Applies `adapt_to_pi` transforms: joint sign flipping and gripper linear-to-angular conversion
  specific to Trossen/Interbotix hardware

None of these apply to YAM. Instead, we use `SimpleDataConfig` with:
- **`YamInputs`**: Maps our camera names directly to model keys, no hardware-specific transforms
- **`YamOutputs`**: Truncates model output to 14 dims, no joint sign un-flipping
- **`DeltaActions`/`AbsoluteActions`**: Applied directly, not wrapped inside Aloha transforms

### Key config details

- **`action_sequence_keys=("action",)`**: Must match the LeRobot dataset key. The default is
  `("actions",)` (plural) — getting this wrong causes a silent data loading failure.
- **`use_quantile_norm`**: Set automatically to `True` for pi0.5 (model type != PI0).
- **Camera name convention**: The model uses "wrist" substring matching for augmentation —
  `left_wrist_0_rgb` and `right_wrist_0_rgb` get color jitter only, while `base_0_rgb` gets
  the full augmentation (crop, rotation, color jitter).

## Training on HPC (Multi-GPU with SLURM)

JAX auto-detects all GPUs on a node — no code changes or `torchrun` needed. With 8 GPUs,
training is ~8x faster (same batch_size=32, 4 samples per GPU via data parallelism).

### 1. Stage data on HPC NFS

The following must be accessible from compute nodes:

| What | Local Path | HPC Action |
|------|-----------|------------|
| openpi repo | `/home/robot/openpi` | `git clone` on login node, then `GIT_LFS_SKIP_SMUDGE=1 uv sync` |
| LeRobot dataset | `~/.cache/huggingface/lerobot/local/yam_simpletest/` | Copy to HPC |
| Norm stats | `openpi/assets/pi05_yam_simpletest_lora/` | Already in repo after running `compute_norm_stats.py` |
| Base model weights | `~/.cache/openpi/openpi-assets/checkpoints/pi05_base/` | Pre-copy (HPC compute nodes may lack internet) |

Copy caches from local machine:

```bash
rsync -avP ~/.cache/huggingface/lerobot/local/yam_simpletest/ hpc:/nfs/path/cache/huggingface/lerobot/local/yam_simpletest/
rsync -avP ~/.cache/openpi/openpi-assets/ hpc:/nfs/path/cache/openpi/openpi-assets/
```

### 2. Setup openpi on HPC

```bash
# On login node
git clone <repo-url> /nfs/path/openpi
cd /nfs/path/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 3. Customize and submit SLURM job

Edit `examples/yam/train_hpc.sbatch` — set `OPENPI_DIR` and `CACHE_DIR` to your HPC paths,
and adjust SLURM directives (partition, GPU type, time limit) for your cluster.

```bash
sbatch examples/yam/train_hpc.sbatch
```

The key environment variables in the SLURM script:

- `HF_HOME` — tells LeRobot/HuggingFace where to find cached datasets
- `OPENPI_DATA_HOME` — tells openpi where to find cached base model weights
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` — allocates 90% of GPU memory to JAX

### 4. Verify

```bash
# Check GPU visibility (should show 8)
srun --gpus=8 uv run python -c "import jax; print(f'Devices: {jax.device_count()}')"

# Check training logs
tail -f slurm-<jobid>.log
```

### How multi-GPU works

JAX creates a device mesh of shape `(num_gpus, 1)` — 8-way data parallelism, no model
sharding (FSDP). Each GPU processes `batch_size / num_gpus = 32 / 8 = 4` samples per step.
Gradients are all-reduced across GPUs automatically. The batch_size (32) must be divisible
by the GPU count.

To use fewer GPUs, set `CUDA_VISIBLE_DEVICES` (e.g., `export CUDA_VISIBLE_DEVICES=0,1,2,3`
for 4 GPUs) or adjust `#SBATCH --gpus=N`.
