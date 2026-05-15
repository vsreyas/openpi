# YAM Combined Dataset — Conversion, Norm Stats, Training

End-to-end recipe for pi0.5 LoRA fine-tuning on a single multi-task YAM
LeRobot dataset (pick-place + arrange-corn-knife + wipe-the-tray), with
optional per-episode filters so the same dataset can serve as multiple
training corpora without duplicating video data.

The dataset spans 35 success-only episodes / 43,078 frames / 3 tasks. Each
episode carries:

- the natural-language `task` string (used by `prompt_from_task=True`)
- a per-frame int64 `orig_traj_id_6` column = the last 6 digits of the
  original teleop episode directory name (e.g.
  `pick-place_20260513_014512` -> `14512`)

Filters at training time select whole episodes by `(task, orig_traj_id_6)`.

## Prerequisites

- Conda env `dsrl_pi0` (has `openpi`, `lerobot==0.3.3`, `h5py`, `av`, jax)
- A6000-class GPU (one is enough; LoRA fits in 48 GB at batch 2)
- Raw teleop data at
  `/data/group_data/maxlab/common_datasets/sreyasv/data/data/data/{pick-place,arrange-corn-knife,wipe-the-tray-with-the-cloth}/`
- pi0.5 base weights at the repo's default cache
  (`/data/group_data/rl/sreyasv/base_dump/openpi-assets/checkpoints/pi05_base/params/`).
  Already populated. Override with `OPENPI_DATA_HOME` if you want elsewhere.

All commands assume:

```bash
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0
cd /home/sreyasv/Projects/dsrl_pi0/openpi
export ROOT=/data/group_data/maxlab/common_datasets/sreyasv/yam_lerobot
export HF_LEROBOT_HOME="$ROOT/lerobot_home"
```

`HF_LEROBOT_HOME` must be exported for **every** step that touches the
LeRobot dataset (conversion, norm stats, training).

## Step 1: Stage success-only symlinks

One subdirectory per task category under a single staging root:

```bash
SRC=/data/group_data/maxlab/common_datasets/sreyasv/data/data/data
STAGE="$ROOT/staging_combined"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{pick-place,arrange-corn-knife,wipe-the-tray-with-the-cloth}
for task in pick-place arrange-corn-knife wipe-the-tray-with-the-cloth; do
    for d in $(ls "$SRC/$task" | grep -v '^FAILED' \
                                 | grep -v 'tar.gz' \
                                 | grep -v 'annotations'); do
        [ -d "$SRC/$task/$d" ] && ln -s "$SRC/$task/$d" "$STAGE/$task/$d"
    done
done
```

Expected counts: pick-place 15, arrange-corn-knife 10, wipe-the-tray 10
(35 total). `FAILED_*` directories are filtered out.

## Step 2: Convert to a single LeRobot dataset

```bash
python examples/yam/convert_yam_combined_to_lerobot.py \
    --staging-root "$ROOT/staging_combined" \
    --repo-id local/yam_combined
```

Runs ~25-30 min on one machine. Writes to
`$HF_LEROBOT_HOME/local/yam_combined/` (~500 MB).

Per-task prompts default to:

| Task subdir | Prompt |
|---|---|
| `pick-place` | put the green block in the right bin and the blue block in the left bin |
| `arrange-corn-knife` | place the knife and the donut on the plate |
| `wipe-the-tray-with-the-cloth` | Wipe the black tray with the white cloth |

Override per task with `--task-prompts pick-place="..." ...`. Re-running
silently `rm -rf`s any existing dataset at the same `--repo-id`.

## Step 3: Compute normalization statistics

Compute once on the unfiltered combined config; the filtered configs reuse
the result via `AssetsConfig`.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/compute_norm_stats.py --config-name pi05_yam_combined_lora
```

Runs ~5 min. Writes
`assets/pi05_yam_combined_lora/local/yam_combined/norm_stats.json`.

## Step 4: Train

Four TrainConfigs are registered. All four point at the same LeRobot
dataset; the filtered ones narrow it at load time. Each runs for
`num_train_steps=20_000` by default.

| Config name | Filter | Frames |
|---|---|---|
| `pi05_yam_combined_lora` | none | 43,078 (35 ep, 3 tasks) |
| `pi05_yam_pickplace_a_lora` | pick-place AND `orig_traj_id_6 == 14512` | 1,097 (1 ep) |
| `pi05_yam_pickplace_b_lora` | pick-place AND `orig_traj_id_6 >= 14512` | 8,923 (8 ep) |
| `pi05_yam_arrange_a_lora` | arrange-corn-knife AND `orig_traj_id_6 == 25632` | 795 (1 ep) |

The smoke pattern (3 steps, batch 2, no wandb) — drop `--num-train-steps`
and `--batch-size` for a real run:

```bash
CKPT_ROOT=/data/group_data/maxlab/common_datasets/sreyasv/vla_project/experiments/openpi_smoke/checkpoints

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_yam_pickplace_a_lora \
    --exp-name=smoke_v2 \
    --num-train-steps=3 \
    --batch-size=2 \
    --num-workers=2 \
    --overwrite \
    --no-wandb-enabled \
    --checkpoint-base-dir="$CKPT_ROOT"
```

For a real fine-tune, drop the smoke flags and run with the config's
defaults (`num_train_steps=20_000`, `batch_size=32`,
`save_interval=5_000`). On 8x A6000s training is ~8x faster (JAX
auto-detects).

Checkpoints land at
`<checkpoint-base-dir>/<config_name>/<exp-name>/<step>/{params,train_state,assets}/`.

## How the load-time filter works

`DataConfig` has four optional fields that drive `_FilteredLeRobotDataset`
in `src/openpi/training/data_loader.py`:

```python
dataset_filter_prompt: str | None
dataset_filter_orig_traj_id_6_eq: int | None
dataset_filter_orig_traj_id_6_min: int | None
dataset_filter_orig_traj_id_6_max: int | None
```

If any are set, the wrapper iterates `meta.episodes`, drops any whose
`tasks` list does not contain the requested prompt or whose first-frame
`orig_traj_id_6` falls outside `[min, max]` / does not equal `eq`. Kept
episodes contribute all their frames; partial episodes are never kept (so
the underlying `delta_timestamps` action-chunking stays valid for every
returned frame).

To add a new filtered config, copy `pi05_yam_pickplace_b_lora` in
`src/openpi/training/config.py`, point its `AssetsConfig` at the same
`./assets/pi05_yam_combined_lora` directory (so it shares norm stats), and
set the four filter fields inside its `base_config=DataConfig(...)`.

## Adding more raw episodes

1. Drop new episode directories into the raw tree under
   `pick-place/`, `arrange-corn-knife/`, `wipe-the-tray-with-the-cloth/`,
   or add a new sibling task directory.
2. Re-run **Step 1** to refresh the symlink tree (or extend the loop with
   the new task name).
3. If you added a new task, pass `--task-prompts <new-task>="..."` on the
   converter command in **Step 2**.
4. Re-run **Step 2** (overwrites the dataset) and **Step 3** (overwrites
   the norm stats).
5. New filtered TrainConfigs (new task / new traj range) require a new
   entry in `src/openpi/training/config.py`.

## Files added / changed by this pipeline

```
openpi/
    examples/yam/
        convert_yam_combined_to_lerobot.py     # new; LeRobot 0.3.3 API
        COMBINED.md                            # this file
    src/openpi/training/
        config.py                              # +3 TrainConfigs, +4 DataConfig filter fields
        data_loader.py                         # +_FilteredLeRobotDataset, +_has_episode_filter
    src/openpi/policies/policy.py              # +return_normalized kwarg on Policy.infer
```
