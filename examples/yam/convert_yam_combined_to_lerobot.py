"""Convert a multi-task YAM teleop tree into a single LeRobot dataset.

Input layout (the staging root produced by symlinks):

    <staging_root>/
        pick-place/
            pick-place_20260513_014512/      <- episode dir (hdf5 + 3 mp4s)
            pick-place_20260513_014606/
            ...
        arrange-corn-knife/
            arrange-corn-knife_20260513_025052/
            ...
        wipe-the-tray-with-the-cloth/
            ...

Each episode dir is the same format as the single-task converter expects:
    episode.hdf5 with actions (N,14), robot/{left,right}/{joint_pos,gripper_pos}
    top.mp4, left_wrist.mp4, right_wrist.mp4 at 60fps.

Output: one LeRobot dataset (e.g. `local/yam_combined`) with:
- `task` per episode set to the prompt for that task category (configurable).
- per-frame int64 column `orig_traj_id_6` = last 6 digits of the original
  episode dirname (e.g. `pick-place_20260513_014512` -> 14512). Used to
  filter at load time without duplicating video data.

This converter is written for LeRobot 0.3.3 (new flat layout, `task` as a
kwarg to add_frame, no support for custom per-episode metadata).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import av
import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import torch
import tqdm
import tyro


IMAGE_SIZE = 224
CAMERAS = ["top", "left_wrist", "right_wrist"]
FPS = 60

# Mapping of task-directory names -> natural-language prompt. Update via CLI
# (--task-prompts pick-place="..." arrange-corn-knife="..." ...).
DEFAULT_TASK_PROMPTS: dict[str, str] = {
    "pick-place": "put the green block in the right bin and the blue block in the left bin",
    "arrange-corn-knife": "place the knife and the donut on the plate",
    "wipe-the-tray-with-the-cloth": "Wipe the black tray with the white cloth",
}


def decode_video_frames(video_path: Path, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode mp4 video and resize to target_size x target_size. (N,H,W,3) uint8."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = np.array(Image.fromarray(img).resize((target_size, target_size), Image.BICUBIC))
        frames.append(img)
    container.close()
    return np.stack(frames)


def parse_traj_id_6(episode_dirname: str) -> int:
    """`pick-place_20260513_014512` -> 14512."""
    last = episode_dirname.rsplit("_", 1)[-1]
    if not last.isdigit() or len(last) != 6:
        raise ValueError(
            f"Episode dir name {episode_dirname!r} does not end with a 6-digit suffix"
        )
    return int(last)


def load_episode(ep_dir: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, int]:
    """Load images (parallel mp4 decode), state, actions, and traj_id_6."""
    with h5py.File(ep_dir / "episode.hdf5", "r") as f:
        left_joint = f["robot/left/joint_pos"][:]
        left_grip = f["robot/left/gripper_pos"][:]
        right_joint = f["robot/right/joint_pos"][:]
        right_grip = f["robot/right/gripper_pos"][:]
        actions = f["actions"][:]

    state = np.concatenate([left_joint, left_grip, right_joint, right_grip], axis=1).astype(np.float32)
    action = actions.astype(np.float32)

    with ThreadPoolExecutor(max_workers=len(CAMERAS)) as pool:
        futures = {cam: pool.submit(decode_video_frames, ep_dir / f"{cam}.mp4") for cam in CAMERAS}
        imgs_per_cam = {cam: fut.result() for cam, fut in futures.items()}

    return imgs_per_cam, torch.from_numpy(state), torch.from_numpy(action), parse_traj_id_6(ep_dir.name)


def collect_combined_episode_dirs(staging_root: Path) -> list[tuple[Path, str]]:
    """Returns [(episode_dir, task_dirname), ...] across all task subdirs."""
    out: list[tuple[Path, str]] = []
    for task_dir in sorted(staging_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() or ep_dir.is_symlink():
                out.append((ep_dir, task_dir.name))
    return out


def create_empty_dataset(repo_id: str, image_size: int = IMAGE_SIZE) -> LeRobotDataset:
    motors = [
        "left_joint_0", "left_joint_1", "left_joint_2",
        "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper",
        "right_joint_0", "right_joint_1", "right_joint_2",
        "right_joint_3", "right_joint_4", "right_joint_5", "right_gripper",
    ]

    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        # Per-frame int64. Same value broadcast across the episode.
        "orig_traj_id_6": {"dtype": "int64", "shape": (1,), "names": ["orig_traj_id_6"]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }

    target = HF_LEROBOT_HOME / repo_id
    if target.exists():
        print(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="yam",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )


def convert(
    staging_root: Path,
    repo_id: str = "local/yam_combined",
    task_prompts: dict[str, str] = DEFAULT_TASK_PROMPTS,
    num_preload_workers: int = 4,
) -> None:
    """Convert the multi-task staging tree to one LeRobot dataset.

    Args:
        staging_root: Path with one subdir per task category, each containing
            episode dirs (or symlinks to them).
        repo_id: LeRobot dataset repo id (will live under HF_LEROBOT_HOME).
        task_prompts: Mapping task-dirname -> prompt string written into the
            LeRobot `task` field for every episode under that subdir.
        num_preload_workers: Background episode preload concurrency.
    """
    staging_root = Path(staging_root)
    episode_dirs = collect_combined_episode_dirs(staging_root)
    print(f"Found {len(episode_dirs)} episodes under {staging_root}")

    seen_tasks = sorted({tn for _, tn in episode_dirs})
    missing = [tn for tn in seen_tasks if tn not in task_prompts]
    if missing:
        raise ValueError(
            f"No prompt provided for task subdir(s) {missing}. "
            f"Pass --task-prompts {missing[0]}=\"<prompt>\" ..."
        )

    print("Task -> prompt mapping in use:")
    for tn in seen_tasks:
        n = sum(1 for _, t in episode_dirs if t == tn)
        print(f"  [{n:3d}] {tn}: {task_prompts[tn]!r}")

    dataset = create_empty_dataset(repo_id)

    with ThreadPoolExecutor(max_workers=num_preload_workers) as pool:
        futures = [pool.submit(load_episode, ep_dir) for ep_dir, _ in episode_dirs]

        for (ep_dir, task_name), future in tqdm.tqdm(
            zip(episode_dirs, futures), total=len(futures), desc="Converting"
        ):
            imgs_per_cam, state, action, traj_id_6 = future.result()
            num_frames = state.shape[0]

            for cam, imgs in imgs_per_cam.items():
                assert imgs.shape[0] == num_frames, (
                    f"Frame count mismatch in {ep_dir.name}: {cam} has {imgs.shape[0]}, "
                    f"expected {num_frames}"
                )

            traj_id_arr = np.array([traj_id_6], dtype=np.int64)
            prompt = task_prompts[task_name]
            for i in range(num_frames):
                frame: dict[str, np.ndarray | torch.Tensor] = {
                    "observation.state": state[i],
                    "action": action[i],
                    "orig_traj_id_6": traj_id_arr,
                }
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]
                dataset.add_frame(frame, task=prompt)

            dataset.save_episode()

    dataset.stop_image_writer()
    print(f"Dataset saved to {HF_LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    tyro.cli(convert)
