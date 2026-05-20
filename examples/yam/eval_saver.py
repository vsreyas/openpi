"""Subscriber that saves YAM policy rollouts to disk.

Modeled after yam_teleop/scripts/collect_data.py's per-episode layout:
one directory per episode containing an HDF5 file with actions / state /
timestamps and one ffmpeg-encoded video file per camera. There is no
GELLO stream and no audio in this path — this is a pure policy rollout.

Observations come from YamEnvironment.get_observation() and have:
    state:  float32 [14]  (left joint(6), left grip(1), right joint(6), right grip(1))
    images: dict[str, uint8 (3, H, W)]  RGB, CHW
    prompt: optional str

Actions come from PolicyAgent.get_action(); expected layout is
{"actions": np.ndarray}, matching YamEnvironment.apply_action().
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

from yam_teleop.video import DEFAULT_PRESET, VIDEO_PRESETS, StreamingVideoWriter


def _chw_rgb_to_hwc_bgr(img: np.ndarray) -> np.ndarray:
    """Convert a CHW RGB uint8 frame to a contiguous HWC BGR frame.

    StreamingVideoWriter pipes raw bytes into ffmpeg with `-pix_fmt bgr24`,
    so frames must be HWC BGR. YamEnvironment emits CHW RGB.
    """
    hwc = np.transpose(img, (1, 2, 0))
    return np.ascontiguousarray(hwc[..., ::-1])


class RolloutSaverSubscriber(_subscriber.Subscriber):
    """Save each episode under `<output_dir>/<run_tag>_<timestamp>/`."""

    def __init__(
        self,
        output_dir: str | Path,
        run_tag: str = "rollout",
        prompt: str | None = None,
        codec: str = DEFAULT_PRESET,
        fps: int = 60,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_tag = run_tag
        self._prompt = prompt
        self._codec_config = VIDEO_PRESETS[codec]
        self._fps = fps

        self._episode_dir: Path | None = None
        self._tmp_dir: Path | None = None
        self._video_writer: StreamingVideoWriter | None = None
        self._buffers: dict | None = None
        self._episode_count = 0
        self._start_wall_time: float = 0.0

    @override
    def on_episode_start(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_dir = self._output_dir / f"{self._run_tag}_{stamp}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        self._tmp_dir = Path(tempfile.mkdtemp(dir=self._episode_dir, prefix=".tmp_video_"))
        self._video_writer = StreamingVideoWriter(
            self._tmp_dir, codec=self._codec_config, fps=self._fps,
        )
        self._buffers = {
            "actions": [],
            "state": [],
            "step_ns": [],
        }
        self._start_wall_time = time.time()
        logging.info("RolloutSaver: episode dir = %s", self._episode_dir)

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        if self._buffers is None or self._video_writer is None:
            raise RuntimeError("on_step called before on_episode_start")

        # --- Scalar / vector channels --------------------------------------
        if "actions" in action:
            act = np.asarray(action["actions"])
        else:
            act = np.asarray(action)
        self._buffers["actions"].append(act.copy())
        self._buffers["state"].append(np.asarray(observation["state"]).copy())
        self._buffers["step_ns"].append(time.time_ns())

        # --- Per-camera frames ---------------------------------------------
        images = observation["images"]
        if not self._video_writer.is_started:
            first = next(iter(images.values()))
            _, h, w = first.shape
            self._video_writer.start(list(images.keys()), w, h)
        for cam_name, frame in images.items():
            self._video_writer.write_frame(cam_name, _chw_rgb_to_hwc_bgr(frame))

    @override
    def on_episode_end(self) -> None:
        if self._buffers is None or self._video_writer is None or self._episode_dir is None:
            return

        n_steps = len(self._buffers["actions"])
        duration_s = time.time() - self._start_wall_time
        logging.info(
            "RolloutSaver: finalizing episode (%d steps, %.1fs wall)",
            n_steps, duration_s,
        )

        # Flush ffmpeg, then move per-camera files into the episode dir.
        video_result = self._video_writer.finish()
        video_info: dict[str, tuple[str, int]] = {}
        for cam_name, (tmp_path, num_frames) in video_result.items():
            safe_name = cam_name.replace("/", "_")
            final_name = f"{safe_name}.{self._codec_config.container}"
            final_path = self._episode_dir / final_name
            shutil.move(str(tmp_path), str(final_path))
            video_info[cam_name] = (final_name, num_frames)

        # HDF5 next to the video files.
        hdf5_path = self._episode_dir / "episode.hdf5"
        self._save_hdf5(hdf5_path, video_info, n_steps)

        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

        self._episode_count += 1
        logging.info("RolloutSaver: saved %s", self._episode_dir)

        self._episode_dir = None
        self._tmp_dir = None
        self._video_writer = None
        self._buffers = None

    def _save_hdf5(
        self,
        path: Path,
        video_info: dict[str, tuple[str, int]],
        n_steps: int,
    ) -> None:
        assert self._buffers is not None
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "actions",
                data=np.array(self._buffers["actions"]),
                compression="gzip",
            )
            f.create_dataset(
                "state",
                data=np.array(self._buffers["state"]),
                compression="gzip",
            )
            ts = f.create_group("timestamps")
            ts.create_dataset("step_ns", data=np.array(self._buffers["step_ns"]))

            images = f.create_group("images")
            for cam_name, (video_filename, num_frames) in video_info.items():
                cam_group = images.create_group(cam_name)
                cam_group.attrs["video_file"] = video_filename
                cam_group.attrs["codec"] = self._codec_config.codec
                cam_group.attrs["num_frames"] = num_frames
                cam_group.attrs["fps"] = self._fps

            f.attrs["num_steps"] = n_steps
            f.attrs["image_storage"] = "video"
            f.attrs["created_at"] = datetime.now().isoformat()
            if self._prompt is not None:
                f.attrs["prompt"] = self._prompt
