# YAM Data Conversion

Converts raw YAM teleop data (HDF5 + mp4) into the LeRobot dataset format used by openpi.

## Prerequisites

- openpi repo cloned and synced:
  ```bash
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  ```
- **Important**: Deactivate conda before running any `uv` commands. openpi uses `uv` for
  dependency management, which manages its own `.venv`. Conda and uv virtualenvs conflict
  with each other, so always run `conda deactivate` first (verify with `echo $CONDA_PREFIX`
  — it should be empty).

## Data Format

Each episode folder should contain:
```
episode_folder/
  episode.hdf5      # robot state + actions
  top.mp4           # top camera (1280x720, 60fps)
  left_wrist.mp4    # left wrist camera
  right_wrist.mp4   # right wrist camera
```

HDF5 fields used:
- `robot/left/joint_pos` (N, 6) + `robot/left/gripper_pos` (N, 1)
- `robot/right/joint_pos` (N, 6) + `robot/right/gripper_pos` (N, 1)
- `actions` (N, 14) - absolute action targets

The state vector is assembled as:
`[left_joint_pos(6), left_gripper(1), right_joint_pos(6), right_gripper(1)]` — 14 dimensions
total, matching the action space.

## Command

```bash
uv run examples/yam/convert_yam_data_to_lerobot.py --raw-dir ../gello/yam_teleop/data/simpletest --repo-id local/yam_simpletest --task "pick up the lego block and place into the box"
```

The dataset is saved to `~/.cache/huggingface/lerobot/local/yam_simpletest/`.

Options:
- `--num-preload-workers 4` — number of episodes to pre-load in parallel (default 4)
- `--push-to-hub` — push to HuggingFace Hub after conversion

## What happens in this step

openpi uses the [LeRobot](https://github.com/huggingface/lerobot) dataset format. Our raw data
(HDF5 + mp4 files) needs to be converted into this format so the openpi data loader can read it.

The conversion script does the following for each episode:

1. **Reads robot state from HDF5**: Loads `robot/{left,right}/joint_pos` and `gripper_pos`,
   concatenates them into a 14-dim state vector.
2. **Reads absolute actions from HDF5**: The `actions` field (14-dim) contains absolute joint
   position targets — where each joint should move to, not how far to move.
3. **Decodes and resizes video frames**: Each mp4 video (1280x720) is decoded frame-by-frame
   using PyAV and resized to 224x224 using PIL bicubic interpolation. This is done at conversion
   time rather than training time to avoid loading large 720p images into the data pipeline
   (the model expects 224x224 anyway). The 3 camera videos per episode are decoded in parallel
   using threads.
4. **Writes to LeRobot format**: Each frame is passed to `LeRobotDataset.add_frame()` which
   stores state/action data in parquet files and re-encodes images into new mp4 videos. This
   means there is a decode-then-re-encode round trip for the videos — LeRobot requires this
   because it controls the output codec and file layout (one mp4 per episode per camera).
5. **Multiple episodes are pre-loaded in parallel** to overlap video decoding with dataset writing.

The resulting LeRobot dataset structure:
```
~/.cache/huggingface/lerobot/local/yam_simpletest/
  data/chunk-000/episode_000000.parquet   # state, action per frame
  data/chunk-000/episode_000001.parquet
  ...
  videos/chunk-000/observation.images.top/episode_000000.mp4
  videos/chunk-000/observation.images.left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
  ...
  meta/info.json          # fps, features, total episodes/frames
  meta/episodes.jsonl     # per-episode metadata
  meta/tasks.jsonl        # task descriptions
```

## Adding More Data

To add new episodes, collect them in a new data folder and re-run conversion with a new repo-id,
or re-run conversion on the combined data folder. Then recompute norm stats and retrain.
