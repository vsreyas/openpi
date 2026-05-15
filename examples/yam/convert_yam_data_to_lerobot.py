"""
Script to convert YAM teleop hdf5+mp4 data to LeRobot dataset format.

Each episode folder is expected to contain:
  - episode.hdf5  (robot state, actions, timestamps)
  - top.mp4, left_wrist.mp4, right_wrist.mp4  (camera videos)

The HDF5 file contains:
  - actions: (N, 14) absolute action targets
  - robot/left/joint_pos: (N, 6), robot/left/gripper_pos: (N, 1)
  - robot/right/joint_pos: (N, 6), robot/right/gripper_pos: (N, 1)

observation.state is assembled as:
  [left_joint_pos(6), left_gripper(1), right_joint_pos(6), right_gripper(1)]

Images are decoded from mp4 and resized to 224x224.

Example usage:
  uv run examples/yam/convert_yam_data_to_lerobot.py \
    --raw-dir ../yam/yam_teleop/data/simpletest \
    --repo-id local/simpletest \
    --task "pick up the lego block and place into the box"

To combine multiple raw_dirs (each containing their own episode dirs) into a
single LeRobot dataset, place them under a parent directory and pass
--multi-source. For example, ../gello/yam_teleop/data/3lego contains 12
sub-raw-dirs (3lego_task1, 3lego_task8, ...), each holding episode dirs:

  uv run examples/yam/convert_yam_data_to_lerobot.py \
    --raw-dir ../gello/yam_teleop/data/3lego \
    --repo-id local/3lego \
    --multi-source
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


def decode_video_frames(video_path: Path, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode mp4 video and resize to target_size x target_size.

    Returns array of shape (N, target_size, target_size, 3) uint8.
    """
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = np.array(Image.fromarray(img).resize((target_size, target_size), Image.BICUBIC))
        frames.append(img)
    container.close()
    return np.stack(frames)


def load_episode(ep_dir: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """Load one episode: images from mp4 (parallel), state and actions from hdf5."""
    # Load state and actions from HDF5.
    with h5py.File(ep_dir / "episode.hdf5", "r") as f:
        left_joint = f["robot/left/joint_pos"][:]  # (N, 6)
        left_grip = f["robot/left/gripper_pos"][:]  # (N, 1)
        right_joint = f["robot/right/joint_pos"][:]  # (N, 6)
        right_grip = f["robot/right/gripper_pos"][:]  # (N, 1)
        actions = f["actions"][:]  # (N, 14)

    state = np.concatenate([left_joint, left_grip, right_joint, right_grip], axis=1).astype(np.float32)
    action = actions.astype(np.float32)

    state = torch.from_numpy(state)
    action = torch.from_numpy(action)

    # Decode all 3 camera videos in parallel.
    with ThreadPoolExecutor(max_workers=len(CAMERAS)) as pool:
        futures = {cam: pool.submit(decode_video_frames, ep_dir / f"{cam}.mp4") for cam in CAMERAS}
        imgs_per_cam = {cam: fut.result() for cam, fut in futures.items()}

    return imgs_per_cam, state, action


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
        image_writer_processes=10,
        image_writer_threads=5,
    )


def convert(
    raw_dir: Path,
    repo_id: str = "local/yam_simpletest",
    task: str = "pick up the lego block and place into the box",
    push_to_hub: bool = False,
    num_preload_workers: int = 4,
    multi_source: bool = False,
):
    """Convert YAM teleop data to LeRobot format.

    Args:
        raw_dir: Path to the directory containing episode folders.
        repo_id: LeRobot dataset repo ID (local or HuggingFace Hub).
        task: Task description for all episodes.
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
        num_preload_workers: Number of episodes to pre-load in parallel.
        multi_source: If True, raw_dir is treated as a directory of raw_dirs
            (each containing episode dirs), and all episodes are combined
            into a single LeRobot dataset.
    """
    raw_dir = Path(raw_dir)
    episode_dirs = collect_episode_dirs(raw_dir, multi_source)
    print(f"Found {len(episode_dirs)} episodes in {raw_dir}")

    dataset = create_empty_dataset(repo_id)

    # Pre-load episodes in background threads while writing the current one.
    with ThreadPoolExecutor(max_workers=num_preload_workers) as pool:
        futures = [pool.submit(load_episode, ep_dir) for ep_dir in episode_dirs]

        for ep_dir, future in tqdm.tqdm(zip(episode_dirs, futures), total=len(futures), desc="Converting"):
            imgs_per_cam, state, action = future.result()
            num_frames = state.shape[0]

            for cam, imgs in imgs_per_cam.items():
                assert imgs.shape[0] == num_frames, (
                    f"Frame count mismatch in {ep_dir.name}: {cam} has {imgs.shape[0]}, expected {num_frames}"
                )

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task,
                }
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]

                dataset.add_frame(frame)

            dataset.save_episode()

    dataset.stop_image_writer()
    print(f"Dataset saved to {HF_LEROBOT_HOME / repo_id}")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert)
