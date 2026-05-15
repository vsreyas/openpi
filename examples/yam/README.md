# YAM Pi0.5 LoRA Pipeline

End-to-end pipeline for fine-tuning pi0.5 on YAM bimanual robot teleop data.

1. **[Data Conversion](DATA_CONVERSION.md)** — Convert HDF5+mp4 teleop data to LeRobot format
2. **[Sub-Task Annotations](SUBTASK_ANNOTATION.md)** — Optional per-segment language annotations for multi-step tasks
3. **[Training](TRAINING.md)** — Compute norm stats and run LoRA fine-tuning
4. **[Evaluation](EVAL.md)** — Run the trained policy on the real robot

## Quick Start

```bash
conda deactivate  # always deactivate conda before using uv

# 1. Convert data
uv run examples/yam/convert_yam_data_to_lerobot.py --raw-dir ../gello/yam_teleop/data/simpletest --repo-id local/yam_simpletest --task "pick up the lego block and place into the box"

# 2. Compute norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_yam_simpletest_lora

# 3. Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam_simpletest_lora --exp-name=yam_lora_v1 --wandb-enabled --overwrite

# 4. Serve policy (Terminal 1)
uv run scripts/serve_policy.py --default-prompt "pick up the lego block and place into the box" policy:checkpoint --policy.config pi05_yam_simpletest_lora --policy.dir checkpoints/pi05_yam_simpletest_lora/yam_lora_v1/<STEP> --policy.repo-id local/yam_simpletest

# 5. Run eval (Terminal 2, gello conda env)
conda activate gello
python examples/yam/eval.py --env-config ../gello/yam_teleop/configs/env.yaml
```

## Sub-Task Conversion (alternative to step 1)

For episodes with per-segment language annotations (`language_annotations.json` in each
episode folder), use the sub-task converter instead. Each annotated segment becomes its
own LeRobot episode with its own task string — no `--task` flag required.

```bash
# Use the raw annotations file
uv run examples/yam/convert_yam_subtask_data_to_lerobot.py \
  --raw-dir ../gello/yam_teleop/data/pack_take_testaudio \
  --repo-id local/yam_subtask

# Use the cleaned/exported annotations (language_annotations.exported.json)
uv run examples/yam/convert_yam_subtask_data_to_lerobot.py \
  --raw-dir ../gello/yam_teleop/data/pack_take_testaudio \
  --repo-id local/yam_subtask \
  --exported
```

Pass `--exported` when the data has been through the cleaning step and you want to read
`language_annotations.exported.json` instead of the raw `language_annotations.json`. See
[SUBTASK_ANNOTATION.md](SUBTASK_ANNOTATION.md) for the annotation format and segmentation
rules.

## Files

| File | Description |
|------|-------------|
| `convert_yam_data_to_lerobot.py` | Data conversion script (single task per episode) |
| `convert_yam_subtask_data_to_lerobot.py` | Data conversion with sub-task language annotations (`--exported` reads the cleaned `language_annotations.exported.json`) |
| `env.py` | Environment wrapper for eval (formats obs for policy server) |
| `eval.py` | Evaluation client script (runs in gello conda env) |
| `src/openpi/policies/yam_policy.py` | YamInputs/YamOutputs transforms |
| `src/openpi/training/config.py` | Training config `pi05_yam_simpletest_lora` |
