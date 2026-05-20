"""Evaluation script for YAM robot with a trained pi0.5 policy.

Connects to an openpi policy server via WebSocket and runs the policy on the real robot.
This script runs in the gello conda environment (separate from the openpi uv environment).

Server (Terminal 1, openpi uv env):
    conda deactivate
    uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_yam_simpletest_lora --policy.dir checkpoints/pi05_yam_simpletest_lora/yam_lora_v1/<STEP> --default-prompt "pick up the lego block and place into the box"

Client (Terminal 2, gello conda env):
    conda activate gello
    python examples/yam/eval.py --env-config ../gello/yam_teleop/configs/env.yaml
"""

import argparse
import math
import logging
import sys
from pathlib import Path

# Allow importing env.py from the same directory when run as `python examples/yam/eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent

import env as _env
from eval_saver import RolloutSaverSubscriber
from yam_teleop.env import YAMBimanualEnv


class SafetyDebugSubscriber(_subscriber.Subscriber):
    """Print robot-node safety diagnostics during policy rollout."""

    def __init__(self, every_steps: int = 30) -> None:
        self._every_steps = max(int(every_steps), 1)
        self._step = 0

    def on_episode_start(self) -> None:
        self._step = 0
        logging.info("SafetyDebug: printing every %d step(s)", self._every_steps)

    @staticmethod
    def _fmt_float(value) -> str:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(x):
            return "n/a"
        return f"{x:.3f}"

    @staticmethod
    def _max_abs(values) -> str:
        if values is None:
            return "n/a"
        try:
            arr = [abs(float(v)) for v in values]
        except (TypeError, ValueError):
            return "n/a"
        return f"{max(arr):.3f}" if arr else "n/a"

    def on_step(self, observation: dict, action: dict) -> None:
        self._step += 1
        if self._step % self._every_steps != 0:
            return

        safety = observation.get("safety") or {}
        if not safety:
            logging.info(
                "SafetyDebug step=%d: no safety diagnostics in observation "
                "(is robot_node safety.enabled true?)",
                self._step,
            )
            return

        parts = []
        for side in ("left", "right"):
            diag = safety.get(side)
            if not diag:
                parts.append(f"{side}: no_diag")
                continue
            active = diag.get("active_constraints") or []
            warnings = diag.get("warnings") or []
            parts.append(
                f"{side}: active={active or '-'} warnings={warnings or '-'} "
                f"qerr={self._max_abs(diag.get('q_error'))} "
                f"self={self._fmt_float(diag.get('min_self_distance'))} "
                f"inter={self._fmt_float(diag.get('min_interarm_distance'))} "
                f"world={self._fmt_float(diag.get('min_world_distance'))}"
            )
        logging.info("SafetyDebug step=%d | %s", self._step, " | ".join(parts))

    def on_episode_end(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate YAM policy")
    parser.add_argument("--env-config", required=True, help="Path to gello env.yaml")
    parser.add_argument("--host", default="0.0.0.0", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--action-horizon", type=int, default=30, help="Steps before re-querying policy (default: 30 = 0.5s at 60Hz)")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-episode-steps", type=int, default=3600, help="Max steps per episode (default: 3600 = 60s at 60Hz)")
    parser.add_argument("--prompt", default=None, help="Override server's default prompt")
    parser.add_argument("--no-reset", action="store_true", help="Do not send the robot home at episode start/end (for continuous prompting)")
    parser.add_argument(
        "--init-traj-glob",
        action="append",
        default=None,
        help=(
            "Glob (repeatable, ** supported) matching training-trajectory episode.hdf5 files. "
            "On each reset, one match is picked at random and the robot is driven to its "
            "`--init-timestep` state after homing."
        ),
    )
    parser.add_argument("--init-timestep", type=int, default=200, help="Timestep within the picked trajectory to use as the reset target (default: 200)")
    parser.add_argument("--init-seed", type=int, default=None, help="RNG seed for selecting which training trajectory to init from")
    parser.add_argument(
        "--init-traj-id-min",
        type=int,
        default=None,
        help=(
            "Lower bound (inclusive) on the trajectory id (last `_`-separated int "
            "in the episode dir name = orig_traj_id_6). Used to narrow --init-traj-glob "
            "to the same training subset the policy was trained on (mirrors "
            "dataset_filter_orig_traj_id_6_min)."
        ),
    )
    parser.add_argument(
        "--init-traj-id-max",
        type=int,
        default=None,
        help="Upper bound (inclusive) on the trajectory id; mirrors dataset_filter_orig_traj_id_6_max.",
    )
    parser.add_argument("--save-rollouts", action="store_true", help="Save each rollout (HDF5 + per-camera videos) to --save-dir")
    parser.add_argument("--save-dir", default=None, help="Directory for saved rollouts; required when --save-rollouts is set")
    parser.add_argument("--save-run-tag", default="rollout", help="Prefix for each saved episode directory (default: rollout)")
    parser.add_argument(
        "--safety-debug-every",
        type=int,
        default=0,
        help="Print robot-node safety diagnostics every N rollout steps; 0 disables.",
    )
    args = parser.parse_args()

    if args.save_rollouts and not args.save_dir:
        parser.error("--save-dir is required when --save-rollouts is set")

    ws_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_policy.get_server_metadata()}")

    yam_env = YAMBimanualEnv(args.env_config)

    subscribers = []
    if args.safety_debug_every > 0:
        subscribers.append(SafetyDebugSubscriber(every_steps=args.safety_debug_every))
    if args.save_rollouts:
        subscribers.append(
            RolloutSaverSubscriber(
                output_dir=args.save_dir,
                run_tag=args.save_run_tag,
                prompt=args.prompt,
            )
        )

    runtime = _runtime.Runtime(
        environment=_env.YamEnvironment(
            yam_env,
            prompt=args.prompt,
            no_reset=args.no_reset,
            init_traj_globs=args.init_traj_glob,
            init_timestep=args.init_timestep,
            init_seed=args.init_seed,
            init_traj_id_min=args.init_traj_id_min,
            init_traj_id_max=args.init_traj_id_max,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=subscribers,
        max_hz=60,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
