from typing import Dict

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs, noise=noise)
            self._cur_step = 0

        results = self._slice_action_chunks(self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def _slice_action_chunks(self, value, key_path: tuple[str, ...] = ()):
        if isinstance(value, dict):
            return {
                key: self._slice_action_chunks(child, (*key_path, str(key)))
                for key, child in value.items()
            }

        is_action_field = any(
            key in {"action", "actions"} or key.endswith("_actions")
            for key in key_path
        )
        if is_action_field and isinstance(value, np.ndarray) and value.ndim > 0:
            if self._cur_step >= value.shape[0]:
                raise ValueError(
                    f"Action field {'/'.join(key_path)!r} has chunk length "
                    f"{value.shape[0]}, shorter than requested step {self._cur_step}."
                )
            return value[self._cur_step, ...]

        return value

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
    
    @override
    def get_prefix_rep(self, observation: Dict) -> Dict:
        return self._policy.get_prefix_rep(observation)
