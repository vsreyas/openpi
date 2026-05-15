# Sub-Task Language Annotations for YAM

## Motivation

Standard YAM data collection assigns a single language instruction to an entire episode (e.g., "pick up the lego block and place into the box"). This works for simple tasks but breaks down for multi-step manipulation where different phases require different instructions.

Sub-task annotations allow operators to segment an episode into semantically meaningful phases, each with its own natural language description. This enables training language-conditioned policies that can follow step-by-step instructions.

## Data Collection Protocol

During teleoperation, the operator uses a USB foot pedal and microphone to annotate sub-tasks:

1. **Start of sub-task**: Step on foot pedal, speak the sub-task description into the microphone
2. **During speech**: Keep foot pedal held down. Do NOT move the robot arms
3. **End of speech**: Release foot pedal, then perform the described sub-task
4. **Repeat** for each subsequent sub-task within the episode

The system records audio files (`.wav`) and generates per-episode `language_annotations.json` with transcribed text and frame-level timestamps.

## Annotation Format

Each episode folder contains a `language_annotations.json` alongside the standard HDF5 + MP4 files:

```
episode_folder/
  episode.hdf5
  top.mp4, left_wrist.mp4, right_wrist.mp4
  language_annotations.json          # <-- new
  audio_1_YYYYMMDD_HHMMSS.wav       # raw audio recordings
  audio_1_YYYYMMDD_HHMMSS.json      # per-audio metadata
  ...
```

`language_annotations.json` schema:
```json
[
  {
    "text": "Put the yellow block inside the box.",
    "start_step": 114,
    "end_step": 281,
    "start_time": 1775524854.41,
    "end_time": 1775524859.12,
    "audio_file": "audio_1_20260406_212059.wav"
  },
  {
    "text": "Put the blue block inside the box.",
    "start_step": 739,
    "end_step": 862,
    "audio_file": "audio_2_20260406_212115.wav"
  }
]
```

- `start_step` / `end_step`: Frame indices when the foot pedal was held (speech window)
- `text`: Transcribed sub-task instruction

## Sub-Task Segmentation

Each sub-task starts at the **beginning of its speech** (`start_step`). Speech frames are included because the operator may still be moving the robot during speech. Frames before the first annotation are skipped (no task assigned).

```
Frame:  0 ---[114===281]------- [739===862]-------- 1423
              speech 1              speech 2
        SKIP  |     SUB-TASK 1      |    SUB-TASK 2     |
              114------------------738  739------------1422
```

Rules:
- **Frames before first `start_step`**: Skipped (no task assigned)
- **Sub-task i frames**: `annotations[i].start_step` to `annotations[i+1].start_step`
- **Last sub-task frames**: `annotations[-1].start_step` to end of episode

Each sub-task segment is written as a **separate LeRobot episode**. This is critical because LeRobot's action chunk sampling (60-frame horizon) only respects episode boundaries -- splitting prevents action chunks from bleeding across sub-task boundaries.

## Conversion

```bash
uv run examples/yam/convert_yam_subtask_data_to_lerobot.py \
  --raw-dir ../gello/yam_teleop/data/pack_take_testaudio \
  --repo-id local/yam_subtask
```

This produces a LeRobot dataset at `~/.cache/huggingface/lerobot/local/yam_subtask/` where:
- Each sub-task is a separate episode
- Each frame's `task_index` maps to the sub-task's language annotation
- `meta/tasks.jsonl` contains all unique sub-task descriptions

### Using cleaned annotations (`--exported`)

If the raw `language_annotations.json` has been through a data-cleaning pass, the cleaned
output is written next to it as `language_annotations.exported.json`. Pass `--exported`
to read the cleaned file instead of the raw one:

```bash
uv run examples/yam/convert_yam_subtask_data_to_lerobot.py \
  --raw-dir ../gello/yam_teleop/data/pack_take_testaudio \
  --repo-id local/yam_subtask \
  --exported
```

The flag only changes which annotations file is read per episode — segmentation rules and
output format are unchanged. Episodes missing the selected file fall back to an empty
task string (with a warning).

## Training

```bash
# 1. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_yam_subtask_lora

# 2. Train with LoRA
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam_subtask_lora \
  --exp-name=yam_subtask_v1 --wandb-enabled --overwrite
```

The `pi05_yam_subtask_lora` config uses `prompt_from_task=True`, which reads per-frame task annotations from the dataset instead of a hardcoded prompt. The model/LoRA/training parameters are otherwise identical to `pi05_yam_simpletest_lora`.

## Evaluation

When serving the trained policy, pass the appropriate sub-task prompt:

```bash
uv run scripts/serve_policy.py \
  --default-prompt "Put the yellow block inside the box." \
  policy:checkpoint \
  --policy.config pi05_yam_subtask_lora \
  --policy.dir checkpoints/pi05_yam_subtask_lora/yam_subtask_v1/STEP
```

For multi-task evaluation where the prompt changes during execution, the eval client should set the `"prompt"` field in the observation dict sent to the policy server for each query.
