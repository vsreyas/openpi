"""Bundled policy server for YAM bimanual residual RL.

Hosts a frozen pi0.5 (via openpi) *and* the jaxrl2 residual actor in a single
process. Exposes:
    - method="infer"               : run base policy + residual actor, return absolute action chunk.
    - method="update_actor_params" : atomically swap the residual actor params (used during training).

Wire-compatible with `examples/yam/eval.py` so the deploy-time client can talk to
this server unchanged. The trainer (examples/train_yam_residual.py) talks to the
same server during training, periodically pushing fresh residual params.

This file is the openpi -> jaxrl2 bridge. openpi does not depend on jaxrl2 in
the package, but this script imports both.
"""

import argparse
import asyncio
import dataclasses
import http
import json
import logging
import os
import pathlib
import socket
import sys
import time
import traceback
from typing import Any

# Allow running from a fresh openpi env without jaxrl2 yet installed: fall back
# to a sys.path bootstrap pointing at the parent dsrl_pi0 repo.
try:  # noqa: SIM105
    import jaxrl2  # noqa: F401
except ImportError:
    _repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_repo_root))

import jax
import jax.numpy as jnp
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from jaxrl2.agents.pixel_sac.pixel_parl_residual_learner import PixelPARLResidualLearner
from jaxrl2.agents.pixel_sac.pixel_sac_residual_learner import PixelSACResidualLearner

logger = logging.getLogger(__name__)


# Algos understood by the bundled server. Maps `variant["algo"]` -> learner class.
_LEARNER_BY_ALGO = {
    "residual_sac": PixelSACResidualLearner,
    "residual_parl": PixelPARLResidualLearner,
}
# SAC-only kwargs PARL's __init__ does not accept; mirrors train_yam_residual._build_learner.
_PARL_DROP_KWARGS = frozenset({
    "temp_lr", "init_temperature", "target_entropy", "backup_entropy",
    "clip_temp", "clip_min_temp", "clip_max_temp",
})


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    base_config: str
    base_dir: str
    residual_args: str
    residual_checkpoint: str | None = None
    base_repo_id: str | None = None
    default_prompt: str | None = None
    port: int = 8000
    mem_fraction: float = 0.7
    host: str = "0.0.0.0"


def _parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True, help="openpi TrainConfig name, e.g. pi05_yam_simpletest_lora")
    p.add_argument("--base-dir", required=True, help="openpi checkpoint directory for pi0.5")
    p.add_argument("--base-repo-id", default=None, help="Optional LeRobot repo_id override for the base config")
    p.add_argument("--residual-args", required=True, help="Path to launch_args.json written by the trainer")
    p.add_argument("--residual-checkpoint", default=None,
                   help="Optional jaxrl2 residual checkpoint dir; placeholder params used if absent")
    p.add_argument("--default-prompt", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--mem-fraction", type=float, default=0.7)
    ns = p.parse_args()
    return Args(
        base_config=ns.base_config,
        base_dir=ns.base_dir,
        base_repo_id=ns.base_repo_id,
        residual_args=ns.residual_args,
        residual_checkpoint=ns.residual_checkpoint,
        default_prompt=ns.default_prompt,
        port=ns.port,
        host=ns.host,
        mem_fraction=ns.mem_fraction,
    )


# -----------------------------------------------------------------------------
# VLM embedding helper
# -----------------------------------------------------------------------------


def _extract_vlm_hidden_state(raw: Any) -> np.ndarray | None:
    """Pull the prefix hidden state out of pi0's `vlm_embedding` payload.

    pi0.sample_actions returns `(x_0, (prefix_hidden_state, kv_cache))` when
    `return_vlm_embedding=True`, and `Policy.infer` forwards that 2-tuple
    verbatim under the `vlm_embedding` key. We only want the hidden state on
    the wire; the kv_cache is JAX-specific scaffolding and not used by the
    residual actor.

    The hidden state arrives as `(1, S, W)`; we drop the batch dim and
    **mean-pool across the sequence axis** to produce a `(W,)` vector. This
    mirrors the sim residual path (see `train_utils_sim_residual.py:761`,
    where every consumer reduces the prefix hidden state to `(W,)` via
    `np.mean(..., axis=0)` before storing it as `(1, W, 1)` in the obs dict).

    Returns a `(hidden_dim,)` array, or `None` if `raw` is falsy.
    """
    if raw is None:
        return None
    # The "tuple" may already be a list/tuple of numpy arrays after the wire-
    # side conversion that happens in some configurations.
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        hidden_state = raw[0]
    else:
        hidden_state = raw
    arr = np.asarray(hidden_state, dtype=np.float32)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]  # drop leading batch dim -> (S, W)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)  # (W,)
    return arr


# -----------------------------------------------------------------------------
# Residual policy wrapper
# -----------------------------------------------------------------------------


class ResidualPolicyWrapper(_base_policy.BasePolicy):
    """Composes a frozen pi0.5 base policy with a jaxrl2 residual actor.

    The residual actor is run with `predict_a_exec=True`: its output is
    a_exec_norm in [-1, 1] normalized space (i.e. before the openpi
    Unnormalize/AbsoluteActions/YamOutputs transform group). a_base_norm is
    fed as conditioning into the actor's observation (`base_action` key) but
    is NOT additively composed with the actor output.

    Args:
        base_policy: the openpi Policy wrapping pi0.5.
        residual_agent: a jaxrl2 PixelSACResidualLearner whose actor params we
            mirror on the server side.
        variant: the trainer's launch-args dict (sets `use_vlm_embedding`,
            `chunk_len`, `query_freq`, `num_cameras`, etc.).
        default_prompt: prompt used when the client doesn't supply one.
    """

    def __init__(
        self,
        base_policy: _policy.Policy,
        residual_agent: PixelSACResidualLearner | PixelPARLResidualLearner,
        variant: dict,
        default_prompt: str | None = None,
    ):
        self._base = base_policy
        self._agent = residual_agent
        self._variant = variant
        self._default_prompt = default_prompt
        self._params_lock = asyncio.Lock()
        self._actor_version = 0
        self._use_vlm_embedding = bool(variant.get("use_vlm_embedding", False))
        self._chunk_len = int(variant.get("chunk_len", 60))
        self._query_freq = int(variant.get("query_freq", 30))
        self._num_cameras = int(variant.get("num_cameras", 3))
        self._action_dim = 14
        # The trainer's actor outputs (query_freq * action_dim) flat; reshape.
        self._actor_chunk_shape = (self._query_freq, self._action_dim)

    # ---- inference path ----

    def _build_base_input(self, obs: dict) -> dict:
        """Pass through to the base policy's expected wire format."""
        # `obs` already matches what ref/openpi/examples/yam/env.py produces:
        # {"images": {"top": chw, "left_wrist": chw, "right_wrist": chw},
        #  "state": (14,), "prompt": str (optional)}
        out = dict(obs)
        if "prompt" not in out and self._default_prompt is not None:
            out["prompt"] = self._default_prompt
        return out

    def _build_residual_obs(
        self, obs: dict, a_base_norm: np.ndarray, vlm_embedding: np.ndarray | None,
    ) -> dict:
        """Construct the dict the jaxrl2 actor expects (with batch dim of 1)."""
        # Images: (H, W, 3) each -> channel-concat to (H, W, 9), then (1, H, W, 9, 1).
        images = obs["images"]
        # Each camera arrives CHW from the eval client; convert to HWC here so
        # we can channel-concat and match the trainer-side wrapper layout.
        def _to_hwc(img):
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            return arr
        top = _to_hwc(images["top"])
        lw = _to_hwc(images["left_wrist"])
        rw = _to_hwc(images["right_wrist"])
        pixels_hwc9 = np.concatenate([top, lw, rw], axis=2).astype(np.uint8)
        pixels = pixels_hwc9[np.newaxis, ..., np.newaxis]

        state = np.asarray(obs["state"], dtype=np.float32)[np.newaxis, ..., np.newaxis]
        base_action = np.asarray(a_base_norm, dtype=np.float32)[np.newaxis, ..., np.newaxis]

        actor_obs: dict[str, Any] = {
            "pixels": pixels,
            "state": state,
            "base_action": base_action,
        }
        if self._use_vlm_embedding and vlm_embedding is not None:
            actor_obs["vlm_embedding"] = np.asarray(vlm_embedding, dtype=np.float32)[np.newaxis, ..., np.newaxis]
        return actor_obs

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        start = time.monotonic()
        base_input = self._build_base_input(obs)
        base_out = self._base.infer(
            base_input,
            return_normalized=True,
            return_vlm_embedding=self._use_vlm_embedding,
        )
        a_base_norm = np.asarray(base_out["actions"])  # (chunk_len, 14) or (action_horizon, action_dim_padded)
        # Truncate any model-side action-dim padding back to 14 dims.
        a_base_norm = a_base_norm[..., : self._action_dim]
        # `Policy.infer` stores the raw `(prefix_hidden_state, kv_cache)` tuple
        # from pi0.sample_actions in `vlm_embedding`. The kv_cache is not
        # msgpack-serializable and not consumed by the residual actor; drop it
        # and forward only the hidden state (matching the sim_residual code path).
        vlm_embedding = _extract_vlm_hidden_state(base_out.get("vlm_embedding")) if self._use_vlm_embedding else None

        actor_obs = self._build_residual_obs(obs, a_base_norm, vlm_embedding)
        # eval_actions returns (B, action_dim_flat); reshape to (query_freq, 14).
        a_exec_flat = self._agent.eval_actions(actor_obs)
        a_exec_norm = np.asarray(a_exec_flat[0]).reshape(self._actor_chunk_shape)

        # Pad/replicate up to the openpi action_horizon so output transforms see
        # the same shape pi0.5 emits.
        if a_exec_norm.shape[0] < a_base_norm.shape[0]:
            pad = np.broadcast_to(a_exec_norm[-1:], (a_base_norm.shape[0] - a_exec_norm.shape[0], self._action_dim))
            a_exec_full = np.concatenate([a_exec_norm, pad], axis=0)
        else:
            a_exec_full = a_exec_norm[: a_base_norm.shape[0]]

        # Pad action_dim back up to whatever the model emitted, so the
        # transform group's joint-vs-gripper boolean masks line up.
        full_action_dim = np.asarray(base_out["actions"]).shape[-1]
        if a_exec_full.shape[-1] < full_action_dim:
            zero_pad = np.zeros((a_exec_full.shape[0], full_action_dim - a_exec_full.shape[-1]), dtype=np.float32)
            a_exec_padded = np.concatenate([a_exec_full, zero_pad], axis=-1)
        else:
            a_exec_padded = a_exec_full

        # Apply the base policy's output transforms (Unnormalize -> AbsoluteActions -> YamOutputs).
        transformed = self._base._output_transform({  # noqa: SLF001
            "state": base_out["state"],
            "actions": a_exec_padded,
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        result = {
            "actions": np.asarray(transformed["actions"], dtype=np.float32),
            "base_action": a_base_norm.astype(np.float32),
            "a_exec_norm": a_exec_full.astype(np.float32),
            "actor_version": self._actor_version,
            "policy_timing": {"infer_ms": elapsed_ms},
        }
        if self._use_vlm_embedding and vlm_embedding is not None:
            result["vlm_embedding"] = np.asarray(vlm_embedding, dtype=np.float32)
        return result

    # ---- weight push path ----

    async def update_actor_params(self, params_pytree: Any) -> dict:
        """Swap the residual actor's params atomically. Caller holds the websocket."""
        current = self._agent._actor.params  # noqa: SLF001
        current_struct = jax.tree_util.tree_structure(current)
        incoming_struct = jax.tree_util.tree_structure(params_pytree)
        if current_struct != incoming_struct:
            raise ValueError(
                f"Residual actor param structure mismatch on push. "
                f"current={current_struct}, incoming={incoming_struct}"
            )
        new_params = jax.tree.map(jnp.asarray, params_pytree)
        async with self._params_lock:
            self._agent._actor = self._agent._actor.replace(params=new_params)  # noqa: SLF001
            self._actor_version += 1
        return {"ok": True, "actor_version": self._actor_version}


# -----------------------------------------------------------------------------
# Websocket server
# -----------------------------------------------------------------------------


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class ResidualWebsocketServer:
    """Websocket server that adds an `update_actor_params` method on top of the
    standard openpi `infer` / `get_prefix_rep` dispatch."""

    def __init__(
        self,
        policy: ResidualPolicyWrapper,
        host: str,
        port: int,
        metadata: dict | None = None,
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time: float | None = None
        while True:
            try:
                start = time.monotonic()
                message = msgpack_numpy.unpackb(await websocket.recv())
                method = message.get("method", "infer")
                if method == "infer":
                    obs = message.get("obs", message)
                    obs.pop("noise", None)
                    result = self._policy.infer(obs)
                elif method == "update_actor_params":
                    params = message["params"]
                    result = await self._policy.update_actor_params(params)
                else:
                    raise ValueError(f"Unknown method: {method}")

                result["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000.0}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                # Coerce arrays to float32 for the wire (matches openpi behavior).
                result = jax.tree.map(
                    lambda x: np.asarray(x).astype(np.float32) if isinstance(x, (np.ndarray, jnp.ndarray)) else x,
                    result,
                )
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start
            except Exception:  # noqa: BLE001
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


def _build_base_policy(args: Args) -> _policy.Policy:
    """Construct the openpi pi0.5 policy from a checkpoint."""
    cfg = _config.get_config(args.base_config)
    if args.base_repo_id is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.base_repo_id))
    return _policy_config.create_trained_policy(cfg, args.base_dir, default_prompt=args.default_prompt)


def _build_residual_agent(variant: dict, variant_path: str):
    """Reconstruct the residual learner with the same hyperparameters the trainer used.

    Dispatches on `variant['algo']` so the server's actor architecture matches
    the trainer's. Without this dispatch, a `residual_parl` trainer would push
    params into a SAC learner and fail the structure check (or silently use the
    wrong actor head).
    """
    chunk_len = int(variant.get("chunk_len", 60))
    query_freq = int(variant.get("query_freq", 30))
    num_cameras = int(variant.get("num_cameras", 3))
    resize_image = int(variant.get("resize_image", 224))
    action_dim = 14

    # Build a representative observation matching the trainer's YamGymEnv obs space.
    sample_obs: dict[str, np.ndarray] = {
        "pixels": np.zeros((1, resize_image, resize_image, 3 * num_cameras, 1), dtype=np.uint8),
        "state": np.zeros((1, action_dim, 1), dtype=np.float32),
        "base_action": np.zeros((1, chunk_len, action_dim, 1), dtype=np.float32),
    }
    if bool(variant.get("use_vlm_embedding", False)):
        # Server mean-pools the prefix hidden state across the sequence axis
        # (see `_extract_vlm_hidden_state`), so the stored obs has shape
        # `(vlm_embedding_dim, 1)` — matches sim's DummyEnvResidual.
        vlm_dim = int(variant.get("vlm_embedding_dim", 2048))
        sample_obs["vlm_embedding"] = np.zeros((1, vlm_dim, 1), dtype=np.float32)
    sample_act = np.zeros((1, query_freq, action_dim), dtype=np.float32)

    # Subset of variant keys the learner accepts. Pulled defensively so unknown
    # fields in the json don't blow up __init__.
    learner_kwargs_keys = (
        "residual_alpha", "actor_lr", "critic_lr", "temp_lr", "decay_steps",
        "hidden_dims", "cnn_features", "cnn_strides", "cnn_padding", "latent_dim",
        "discount", "tau", "critic_reduction", "dropout_rate", "encoder_type",
        "encoder_norm", "color_jitter", "use_spatial_softmax", "softmax_temperature",
        "aug_next", "use_bottleneck", "init_temperature", "num_qs", "target_entropy",
        "action_magnitude", "num_cameras", "backup_entropy", "critic_pop_base_actions",
        "actor_pop_base_actions", "clip_temp", "clip_min_temp", "clip_max_temp",
        "use_huber_loss", "huber_delta", "max_grad_norm", "num_critic_updates",
        "num_actor_updates", "bc_reg_coeff", "bc_on_success_only", "predict_a_exec",
        "use_vlm_embedding", "freeze_vision_encoder", "log_std_min", "log_std_max",
        "learn_std", "fixed_log_std", "actor_arch", "critic_arch",
        "mip_t_star", "mip_noise_std", "mip_use_film", "mip_q_noise_scale",
    )
    learner_kwargs = {k: variant[k] for k in learner_kwargs_keys if k in variant}
    # YAM defaults if the variant didn't pin them.
    learner_kwargs.setdefault("predict_a_exec", True)
    learner_kwargs.setdefault("num_cameras", num_cameras)

    algo = str(variant.get("algo", "residual_sac"))
    if algo not in _LEARNER_BY_ALGO:
        raise ValueError(
            f"Unsupported algo '{algo}' in {variant_path}. Server supports: "
            f"{sorted(_LEARNER_BY_ALGO)}."
        )
    learner_cls = _LEARNER_BY_ALGO[algo]
    if algo == "residual_parl":
        # PARL has no temperature head; drop SAC-only kwargs (mirrors trainer dispatch).
        learner_kwargs = {k: v for k, v in learner_kwargs.items() if k not in _PARL_DROP_KWARGS}
        # PARL-specific knobs the trainer passes alongside the shared ones.
        for k in ("parl_num_samples", "parl_num_elites", "parl_num_grad_steps", "parl_step_size"):
            if k in variant:
                learner_kwargs[k] = variant[k]

    seed = int(variant.get("seed", 0))
    logger.info(
        "Building %s residual learner from %s with keys: %s",
        algo, variant_path, sorted(learner_kwargs),
    )
    return learner_cls(seed=seed, observations=sample_obs, actions=sample_act, **learner_kwargs)


def _load_variant(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()

    # Apply VRAM budget before any JAX work.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.mem_fraction:.2f}"

    variant = _load_variant(args.residual_args)
    logger.info("Loading base policy (config=%s, dir=%s)", args.base_config, args.base_dir)
    base_policy = _build_base_policy(args)

    logger.info("Building residual learner...")
    agent = _build_residual_agent(variant, args.residual_args)
    if args.residual_checkpoint:
        logger.info("Restoring residual params from %s", args.residual_checkpoint)
        agent.restore_checkpoint(args.residual_checkpoint)
    else:
        logger.info("No --residual-checkpoint provided; using freshly-initialized residual params")
        # Robot-safety guardrail: with predict_a_exec=True the residual output is
        # used as the absolute joint command. Untrained Gaussian actor output is
        # near-random noise; do not let that reach the YAM without a BC warmup
        # first. The trainer ordinarily does BC then pushes warmed params, but
        # if `bc_warmup_steps=0` the first push will be random noise. Shout.
        bc_warmup_steps = int(variant.get("bc_warmup_steps", 0) or 0)
        if bc_warmup_steps <= 0:
            logger.warning(
                "=" * 78 + "\n"
                "  SAFETY WARNING: residual server starting with NO checkpoint AND\n"
                "  the trainer's launch_args.json has bc_warmup_steps=%d.\n"
                "  predict_a_exec=True will run a freshly-initialized actor whose\n"
                "  output is used DIRECTLY as absolute joint commands. Do NOT\n"
                "  connect to a real YAM until BC warmup has run and the trainer\n"
                "  has pushed warmed params (server logs `actor_version` >= 1).\n"
                + "=" * 78,
                bc_warmup_steps,
            )

    wrapper = ResidualPolicyWrapper(
        base_policy=base_policy,
        residual_agent=agent,
        variant=variant,
        default_prompt=args.default_prompt,
    )

    metadata = dict(base_policy.metadata)
    metadata["residual_server"] = True
    metadata["actor_version"] = wrapper._actor_version  # noqa: SLF001

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logger.info("Creating residual server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = ResidualWebsocketServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
