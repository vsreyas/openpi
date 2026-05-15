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
import logging
import sys
from pathlib import Path

# Allow importing env.py from the same directory when run as `python examples/yam/eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

import env as _env
from yam_teleop.env import YAMBimanualEnv


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
    args = parser.parse_args()

    ws_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_policy.get_server_metadata()}")

    yam_env = YAMBimanualEnv(args.env_config)

    runtime = _runtime.Runtime(
        environment=_env.YamEnvironment(yam_env, prompt=args.prompt, no_reset=args.no_reset),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=60,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
