"""YAM environment wrapper for policy evaluation.

Wraps YAMBimanualEnv to implement the openpi-client Environment interface.
Handles observation formatting: state assembly, image resize, and HWC-to-CHW conversion.
"""

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from yam_teleop.env import YAMBimanualEnv


CAMERAS = ["top", "left_wrist", "right_wrist"]


class YamEnvironment(_environment.Environment):
    """Environment wrapper for YAM bimanual robot."""

    def __init__(
        self,
        env: YAMBimanualEnv,
        render_height: int = 224,
        render_width: int = 224,
        prompt: str | None = None,
        no_reset: bool = False,
    ) -> None:
        self._env = env
        self._render_height = render_height
        self._render_width = render_width
        self._prompt = prompt
        self._no_reset = no_reset

    @override
    def reset(self) -> None:
        if self._no_reset:
            return
        self._env.reset()

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

        # Resize and rearrange images to (C, H, W) uint8.
        images = {}
        for cam in CAMERAS:
            img = obs["images"][cam]
            img = image_tools.resize_with_pad(img, self._render_height, self._render_width)
            img = image_tools.convert_to_uint8(img)
            images[cam] = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        out = {"state": state, "images": images}
        if self._prompt is not None:
            out["prompt"] = self._prompt
        return out

    @override
    def apply_action(self, action: dict) -> None:
        self._env.step(action["actions"])
