"""
GPU-accelerated, low-memory variant of convert_yam_subtask_data_to_lerobot.py.

Same input format and segmentation logic. mp4 decoding and 224x224 resize run
on the GPU via ffmpeg with NVDEC hardware decode (`-hwaccel cuda
-hwaccel_output_format cuda`) and the `scale_cuda` filter. Frames are
*streamed* from ffmpeg's stdout one at a time and passed straight to
LeRobot's `add_frame`, instead of being decoded into a per-episode numpy
buffer (~8 GB per episode). This keeps working-set memory in the hundreds of
MB and avoids systemd-oomd kills during sub-task save_episode under memory
pressure.

Requirements:
  - CUDA-capable GPU
  - ffmpeg with cuvid/NVDEC support (verify with: ffmpeg -hwaccels | grep cuda)

Each episode folder is expected to contain:
  - episode.hdf5  (robot state, actions, timestamps)
  - top.mp4, left_wrist.mp4, right_wrist.mp4  (camera videos)
  - language_annotations.json  (sub-task text + frame boundaries)

Sub-task segmentation:
  - The first sub-task runs from frame 0 to the second annotation's start_step.
  - Subsequent sub-tasks run from their annotation's start_step to the next
    annotation's start_step (or end of episode for the last sub-task).
  - Speech frames are included (operator may still be moving during speech).
  - Annotations with text "[problematic]" are dropped along with the frames
    they cover.
  - Episodes with no annotations use empty string as task.
  - Each sub-task becomes a separate LeRobot episode so that action chunk
    sampling respects sub-task boundaries.

Example usage:
  uv run examples/yam/convert_yam_subtask_data_to_lerobot_gpu.py \
    --raw-dir ../gello/yam_teleop/data/pack_take_testaudio \
    --repo-id local/yam_subtask

To combine multiple raw_dirs (each containing their own episode dirs) into a
single LeRobot dataset, place them under a parent directory and pass
--multi-source. For example, ../gello/yam_teleop/data/3lego contains 12
sub-raw-dirs (3lego_task1, 3lego_task8, ...), each holding episode dirs:

  uv run examples/yam/convert_yam_subtask_data_to_lerobot_gpu.py \
    --raw-dir ../gello/yam_teleop/data/3lego \
    --repo-id local/3lego_subtask \
    --multi-source
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

IMAGE_SIZE = 224
CAMERAS = ["top", "left_wrist", "right_wrist"]
FPS = 60


@dataclasses.dataclass
class SubtaskSegment:
    """A segment of an episode corresponding to one sub-task."""

    task_text: str
    start_frame: int  # inclusive
    end_frame: int  # exclusive


def stream_decode_video(video_path: Path, target_size: int = IMAGE_SIZE):
    """Stream frames from mp4 via NVDEC + scale_cuda, yielding one (H,W,3) uint8 frame at a time.

    Avoids buffering the entire decoded video in main-process memory. The
    caller is expected to either consume to completion or .close() the
    generator; the finally block sends SIGPIPE/SIGKILL to ffmpeg as needed.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(video_path),
        "-vf",
        f"scale_cuda={target_size}:{target_size}:format=yuv420p,hwdownload,format=yuv420p",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = target_size * target_size * 3
    try:
        while True:
            data = proc.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(target_size, target_size, 3).copy()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def load_state_action(ep_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load only state and actions from hdf5; return (state, action, num_frames).

    No video is decoded here — videos are streamed lazily during conversion.
    """
    with h5py.File(ep_dir / "episode.hdf5", "r") as f:
        left_joint = f["robot/left/joint_pos"][:]  # (N, 6)
        left_grip = f["robot/left/gripper_pos"][:]  # (N, 1)
        right_joint = f["robot/right/joint_pos"][:]  # (N, 6)
        right_grip = f["robot/right/gripper_pos"][:]  # (N, 1)
        actions = f["actions"][:]  # (N, 14)

    state = np.concatenate([left_joint, left_grip, right_joint, right_grip], axis=1).astype(np.float32)
    action = actions.astype(np.float32)

    return torch.from_numpy(state), torch.from_numpy(action), state.shape[0]


def parse_subtask_segments(annotations_path: Path, episode_length: int) -> list[SubtaskSegment]:
    """Parse language_annotations.json and compute sub-task frame ranges.

    Each sub-task runs from its speech start (start_step) to the next annotation's
    start_step, or end of episode for the last sub-task. Speech frames are included
    because the operator may still be moving the robot during speech.

    Annotations with text "[problematic]" are dropped, and the frames they would
    have covered are excluded from the output (not absorbed into adjacent segments).
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    if not annotations:
        return [SubtaskSegment(task_text="", start_frame=0, end_frame=episode_length)]

    # Sort by start_step defensively.
    annotations = sorted(annotations, key=lambda a: a["start_step"])

    segments = []
    for i, ann in enumerate(annotations):
        start_frame = 0 if i == 0 else ann["start_step"]

        if start_frame >= episode_length:
            print(f"  WARNING: annotation {i} start_step ({start_frame}) >= episode length ({episode_length}), skipping")
            continue

        end_frame = annotations[i + 1]["start_step"] if i < len(annotations) - 1 else episode_length

        end_frame = min(end_frame, episode_length)

        if ann["text"] == "[problematic]":
            print(f"  Dropping [problematic] frames {start_frame}:{end_frame}")
            continue

        if start_frame >= end_frame:
            print(
                f"  WARNING: annotation {i} has no action frames "
                f"(start_frame={start_frame}, end_frame={end_frame}), skipping"
            )
            continue

        segments.append(SubtaskSegment(task_text=ann["text"], start_frame=start_frame, end_frame=end_frame))

    return segments


def collect_episode_dirs(raw_dir: Path, multi_source: bool) -> list[Path]:
    """Collect episode directories from raw_dir.

    If multi_source is False, raw_dir's immediate subdirectories are treated
    as episode dirs. If True, raw_dir is treated as a directory of raw_dirs
    (e.g. one per task), and episode dirs are gathered from each one.
    """
    subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not multi_source:
        return subdirs

    episode_dirs = []
    for sub in subdirs:
        episode_dirs.extend(sorted([d for d in sub.iterdir() if d.is_dir()]))
    return episode_dirs


def create_empty_dataset(repo_id: str, image_size: int = IMAGE_SIZE) -> LeRobotDataset:
    """Create an empty LeRobot dataset with the right feature definitions."""
    motors = [
        "left_joint_0",
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_gripper",
        "right_joint_0",
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="yam",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
    )


def convert(
    raw_dir: Path,
    repo_id: str = "local/yam_subtask",
    push_to_hub: bool = False,
    exported: bool = False,
    multi_source: bool = False,
):
    """Convert YAM teleop data with sub-task annotations to LeRobot format (GPU, streaming).

    Args:
        raw_dir: Path to the directory containing episode folders.
        repo_id: LeRobot dataset repo ID (local or HuggingFace Hub).
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
        exported: If True, use language_annotations.exported.json (cleaned) instead of language_annotations.json.
        multi_source: If True, raw_dir is treated as a directory of raw_dirs
            (each containing episode dirs), and all episodes are combined
            into a single LeRobot dataset.
    """
    annotations_filename = "language_annotations.exported.json" if exported else "language_annotations.json"
    raw_dir = Path(raw_dir)
    episode_dirs = collect_episode_dirs(raw_dir, multi_source)
    print(f"Found {len(episode_dirs)} episodes in {raw_dir}")

    dataset = create_empty_dataset(repo_id)

    subtask_count = 0
    no_annotation_episodes = []

    for ep_dir in tqdm.tqdm(episode_dirs, desc="Converting"):
        state, action, num_frames = load_state_action(ep_dir)

        annotations_path = ep_dir / annotations_filename
        if not annotations_path.exists():
            print(f"  WARNING: {ep_dir.name} has NO {annotations_filename} -- using empty string as task")
            no_annotation_episodes.append(ep_dir.name)
            segments = [SubtaskSegment(task_text="", start_frame=0, end_frame=num_frames)]
        else:
            segments = parse_subtask_segments(annotations_path, num_frames)
            if all(seg.task_text == "" for seg in segments):
                print(f"  WARNING: {ep_dir.name} has NO language annotations -- using empty string as task")
                no_annotation_episodes.append(ep_dir.name)

        # Map every frame index to its sub-task (or None for [problematic]/dropped frames).
        frame_to_seg: list = [None] * num_frames
        for seg in segments:
            for i in range(seg.start_frame, seg.end_frame):
                if 0 <= i < num_frames:
                    frame_to_seg[i] = seg

        # Open one streaming decoder per camera; they run in parallel ffmpeg subprocesses.
        gens = {cam: stream_decode_video(ep_dir / f"{cam}.mp4") for cam in CAMERAS}

        current_seg = None
        ep_subtasks = 0
        try:
            for i in range(num_frames):
                try:
                    cam_imgs = {cam: next(gens[cam]) for cam in CAMERAS}
                except StopIteration:
                    print(f"  WARNING: {ep_dir.name} video stream ended early at frame {i}/{num_frames}")
                    break

                seg = frame_to_seg[i]
                if seg is None:
                    continue
                if seg is not current_seg:
                    if current_seg is not None:
                        dataset.save_episode()
                        subtask_count += 1
                        ep_subtasks += 1
                    current_seg = seg

                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": seg.task_text,
                }
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = cam_imgs[cam]
                dataset.add_frame(frame)

            if current_seg is not None:
                dataset.save_episode()
                subtask_count += 1
                ep_subtasks += 1
        finally:
            for gen in gens.values():
                gen.close()

        print(f"  {ep_dir.name}: {ep_subtasks} sub-tasks written")

    dataset.stop_image_writer()
    print(f"\nDone. {subtask_count} sub-task episodes written.")
    print(f"Dataset saved to {HF_LEROBOT_HOME / repo_id}")

    if no_annotation_episodes:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(no_annotation_episodes)} episode(s) had NO language annotations!")
        print("These episodes were saved with empty string as task:")
        for name in no_annotation_episodes:
            print(f"  - {name}")
        print(f"{'='*60}")

    if push_to_hub:
        dataset.push_to_hub()


def _install_signal_diagnostics():
    """Log who/what kills us so we can debug silent terminations."""

    def _handler(signum, frame):
        name = signal.Signals(signum).name
        try:
            ppid = os.getppid()
            with open(f"/proc/{ppid}/comm") as f:
                pcomm = f.read().strip()
        except Exception:
            pcomm = "?"
        print(
            f"\n[SIGNAL] received {name} (signum={signum}) pid={os.getpid()} ppid={ppid} ({pcomm})",
            flush=True,
        )
        traceback.print_stack(frame)
        sys.stdout.flush()
        sys.stderr.flush()
        # Re-raise default behavior so the process actually exits.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sig, _handler)


if __name__ == "__main__":
    _install_signal_diagnostics()
    tyro.cli(convert)
