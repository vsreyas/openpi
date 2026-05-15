import dataclasses
import enum
import logging
import socket
from typing import Any

from openpi_client import base_policy as _base_policy
import tyro
from typing_extensions import override

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class PolicyPromptLogger(_base_policy.BasePolicy):
    """Logs the prompt fed into the model on each inference call."""

    def __init__(self, policy: _base_policy.BasePolicy, default_prompt: str | None = None):
        self._policy = policy
        self._default_prompt = default_prompt
        self._step = 0

    @override
    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[misc]
        # WebsocketPolicyServer always invokes infer with `noise=<value or None>`
        # (see openpi/serving/websocket_policy_server.py). Forward whatever the
        # caller passes so the wrapper is transparent to the policy below.
        if "prompt" in obs:
            prompt = obs["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            source = "client"
        else:
            prompt = self._default_prompt
            source = "default"
        logging.info("[debug step=%d] prompt (%s): %r", self._step, source, prompt)
        self._step += 1
        return self._policy.infer(obs, **kwargs)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Optional override for the data repo_id. Use this when the checkpoint was trained
    # with a different repo_id than what the config specifies.
    repo_id: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Log the prompt fed into the model on each inference step.
    debug: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero_finetuned",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    # Note: These defaults should be updated as and when new checkpoints are trained. All experiments must be on onboarded here.
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            config = _config.get_config(args.policy.config)
            if args.policy.repo_id is not None:
                config = dataclasses.replace(
                    config, data=dataclasses.replace(config.data, repo_id=args.policy.repo_id)
                )
            return _policy_config.create_trained_policy(
                config, args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    if args.debug:
        policy = PolicyPromptLogger(policy, default_prompt=args.default_prompt)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
