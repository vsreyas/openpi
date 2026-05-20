"""YAM environment wrapper for policy evaluation.

Wraps YAMBimanualEnv to implement the openpi-client Environment interface.
Handles observation formatting: state assembly, BGR-to-RGB conversion,
image resize, and HWC-to-CHW conversion.
"""

import glob as _glob
import logging
import random
from pathlib import Path

import h5py
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from PIL import Image
from typing_extensions import override

from yam_teleop.env import YAMBimanualEnv


CAMERAS = ["top", "left_wrist", "right_wrist"]


def _load_state_at(hdf5_path: str, timestep: int) -> np.ndarray:
    """Read the 14-dim [L joint(6), L grip(1), R joint(6), R grip(1)] vector at `timestep`."""
    with h5py.File(hdf5_path, "r") as f:
        num_steps = int(f.attrs.get("num_steps", f["robot/left/joint_pos"].shape[0]))
        if timestep >= num_steps:
            raise ValueError(
                f"init_timestep={timestep} is out of range for {hdf5_path} "
                f"(num_steps={num_steps})"
            )
        state = np.concatenate([
            f["robot/left/joint_pos"][timestep],
            f["robot/left/gripper_pos"][timestep],
            f["robot/right/joint_pos"][timestep],
            f["robot/right/gripper_pos"][timestep],
        ]).astype(np.float64)
    return state


def _resize_like_training(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[:2] == (height, width):
        return img
    return np.asarray(Image.fromarray(img).resize((width, height), Image.BICUBIC))


class YamEnvironment(_environment.Environment):
    """Environment wrapper for YAM bimanual robot."""

    def __init__(
        self,
        env: YAMBimanualEnv,
        render_height: int = 224,
        render_width: int = 224,
        prompt: str | None = None,
        no_reset: bool = False,
        init_traj_globs: list[str] | None = None,
        init_timestep: int = 200,
        init_seed: int | None = None,
        init_traj_id_min: int | None = None,
        init_traj_id_max: int | None = None,
    ) -> None:
        self._env = env
        self._render_height = render_height
        self._render_width = render_width
        self._prompt = prompt
        self._no_reset = no_reset
        self._init_timestep = init_timestep
        self._init_traj_id_min = init_traj_id_min
        self._init_traj_id_max = init_traj_id_max
        all_matches = self._expand_globs(init_traj_globs)
        self._init_traj_files = self._filter_by_traj_id(
            all_matches, init_traj_id_min, init_traj_id_max
        )
        self._init_rng = random.Random(init_seed)
        if init_traj_globs and not self._init_traj_files:
            raise FileNotFoundError(
                f"--init-traj-glob matched {len(all_matches)} file(s) but none "
                f"survived traj-id filter [min={init_traj_id_min}, max={init_traj_id_max}]: "
                f"{init_traj_globs}"
            )

    @staticmethod
    def _expand_globs(patterns: list[str] | None) -> list[str]:
        if not patterns:
            return []
        matches: list[str] = []
        for pat in patterns:
            matches.extend(sorted(_glob.glob(pat, recursive=True)))
        return [m for m in matches if Path(m).is_file()]

    @staticmethod
    def _traj_id_from_path(path: str) -> int | None:
        """Return the int parsed from the last `_`-separated chunk of the episode
        directory name (= `orig_traj_id_6` column in the combined LeRobot dataset).
        Returns None if the dir name does not end in an integer."""
        dir_name = Path(path).parent.name
        tail = dir_name.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return None

    @classmethod
    def _filter_by_traj_id(
        cls,
        files: list[str],
        tid_min: int | None,
        tid_max: int | None,
    ) -> list[str]:
        if tid_min is None and tid_max is None:
            return files
        kept: list[str] = []
        for f in files:
            tid = cls._traj_id_from_path(f)
            if tid is None:
                continue
            if tid_min is not None and tid < tid_min:
                continue
            if tid_max is not None and tid > tid_max:
                continue
            kept.append(f)
        return kept

    _JOINT_LABELS = (
        "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
        "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
    )

    @staticmethod
    def _format_vec(v: np.ndarray) -> str:
        return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"

    @classmethod
    def _format_delta(cls, current: np.ndarray, target: np.ndarray) -> str:
        delta = target - current
        worst = int(np.argmax(np.abs(delta)))
        per_joint = ", ".join(
            f"{cls._JOINT_LABELS[i]}={delta[i]:+.3f}" for i in range(len(delta))
        )
        return (
            f"  max|delta|={np.max(np.abs(delta)):.3f} rad on "
            f"{cls._JOINT_LABELS[worst]} (cur={current[worst]:+.3f}, "
            f"tgt={target[worst]:+.3f})\n  per-joint delta: {per_joint}"
        )

    def _log_safety_diag(self, prefix: str) -> None:
        obs = self._env._get_obs()
        for side in ("left", "right"):
            safety = obs["robot"].get(f"{side}/safety")
            if safety is None:
                logging.info("%s %s/safety: <not published>", prefix, side)
            else:
                logging.info("%s %s/safety: %s", prefix, side, safety)

    def _drive_to_init_state(self) -> None:
        """After homing, pick a random training trajectory and drive the arms to its `init_timestep` state."""
        if not self._init_traj_files:
            return

        path = self._init_rng.choice(self._init_traj_files)
        target = _load_state_at(path, self._init_timestep)
        current = self._env._get_current_joints(self._env._get_obs())

        logging.info("Init-state target loaded from %s @ t=%d", path, self._init_timestep)
        logging.info("  current: %s", self._format_vec(current))
        logging.info("  target:  %s", self._format_vec(target))
        logging.info("%s", self._format_delta(current, target))
        self._log_safety_diag("BEFORE drive:")

        self._env._move_to(current, target, "Driving to init state...")
        after_move = self._env._get_current_joints(self._env._get_obs())
        moved = after_move - current
        logging.info(
            "After _move_to: max|moved|=%.3f rad on %s (moved=%+.3f, target_move=%+.3f)",
            float(np.max(np.abs(moved))),
            self._JOINT_LABELS[int(np.argmax(np.abs(target - current)))],
            float(moved[int(np.argmax(np.abs(target - current)))]),
            float((target - current)[int(np.argmax(np.abs(target - current)))]),
        )
        logging.info("  current after move: %s", self._format_vec(after_move))
        self._log_safety_diag("AFTER move:")

        self._env._hold_until_converged(target)
        after_hold = self._env._get_current_joints(self._env._get_obs())
        residual = target - after_hold
        worst = int(np.argmax(np.abs(residual)))
        logging.info(
            "After _hold_until_converged: residual max|err|=%.3f rad on %s (cur=%+.3f, tgt=%+.3f)",
            float(np.max(np.abs(residual))),
            self._JOINT_LABELS[worst],
            float(after_hold[worst]),
            float(target[worst]),
        )
        logging.info("  current after hold: %s", self._format_vec(after_hold))
        self._log_safety_diag("AFTER hold:")

    @override
    def reset(self) -> None:
        if self._no_reset:
            return
        self._env.reset()
        self._drive_to_init_state()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        obs = self._env._get_obs()

        # Assemble state: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
        state = np.concatenate([
            obs["robot"]["left/joint_pos"],
            obs["robot"]["left/gripper_pos"],
            obs["robot"]["right/joint_pos"],
            obs["robot"]["right/gripper_pos"],
        ]).astype(np.float32)

        # yam_teleop publishes live OpenCV frames in BGR. Training data is
        # decoded from recorded videos as RGB, then directly resized to 224x224.
        images = {}
        for cam in CAMERAS:
            img = np.ascontiguousarray(obs["images"][cam][..., ::-1])
            img = image_tools.convert_to_uint8(img)
            img = _resize_like_training(img, self._render_height, self._render_width)
            images[cam] = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        out = {"state": state, "images": images}
        if self._prompt is not None:
            out["prompt"] = self._prompt
        return out

    @override
    def apply_action(self, action: dict) -> None:
        self._env.step(action["actions"])
