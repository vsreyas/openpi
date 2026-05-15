"""Data transforms for YAM robot policy.

Maps YAM camera names (top, left_wrist, right_wrist) to model image keys
(base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb) and handles action truncation.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """Parse image from LeRobot format (torch float32, C,H,W) to numpy uint8 (H,W,C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs transform for YAM robot.

    Maps camera names to model-expected keys and passes through state/actions.
    """

    CAMERA_MAP = {
        "top": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]

        images = {}
        image_masks = {}
        for source, dest in self.CAMERA_MAP.items():
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                raise ValueError(f"Expected camera '{source}' not found. Got: {tuple(in_images)}")

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs transform for YAM robot.

    Truncates action output to 14 dims (6 joints + 1 gripper per arm).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
