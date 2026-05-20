# YAM Policy Evaluation

Runs a trained pi0.5 LoRA policy on the real YAM robot.
Requires a trained checkpoint (see [TRAINING.md](TRAINING.md)).

## Architecture: Server + Client

Evaluation requires **two separate terminals** because the openpi policy server (JAX, `uv` env)
and the YAM robot client (gello, `conda` env) have incompatible Python dependencies and cannot
share a process. They communicate over WebSocket.

```
Terminal 1 (openpi uv env)              Terminal 2 (gello conda env)
──────────────────────────              ────────────────────────────
serve_policy.py                         eval.py
  - Loads pi0.5 + LoRA weights            - Creates YAMBimanualEnv
  - Runs inference on GPU                  - Reads robot observations
  - Serves actions via WebSocket           - Sends obs to server
                                           - Executes returned actions
```

## One-Time Setup: Install openpi-client in gello conda env

The `openpi-client` package is a lightweight client library (no JAX/PyTorch) that handles
WebSocket communication and image utilities.

```bash
conda activate gello
pip install /home/robot/openpi/packages/openpi-client/
```

## Terminal 1: Start Policy Server (openpi uv env)

```bash
conda deactivate
cd /home/robot/openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_yam_simpletest_lora --policy.dir checkpoints/pi05_yam_simpletest_lora/yam_lora_v1/<STEP> --default-prompt "pick up the lego block and place into the box"
```

Replace `<STEP>` with the checkpoint step number (e.g., `5000` or `10000`). The server loads
the model, downloads base weights if needed (first run), and listens on port 8000.

## Terminal 2: Run Evaluation Client (gello conda env)

```bash
conda activate gello
cd /home/robot/openpi
python examples/yam/eval.py --env-config ../gello/yam_teleop/configs/env.yaml
```

Options:
- `--host 0.0.0.0` — server host (default: localhost)
- `--port 8000` — server port (default: 8000)
- `--action-horizon 30` — re-query policy every N steps (default: 30 = 0.5s at 60Hz)
- `--num-episodes 1` — number of episodes to run (default: 1)
- `--max-episode-steps 3600` — max steps per episode (default: 3600 = 60s at 60Hz)

## What Happens During Evaluation

1. The client connects to the server via WebSocket and resets the robot to home position.
2. Each episode runs a closed-loop control cycle at 60Hz:
   - **Client** reads observations from the robot (joint positions + camera images).
   - **Client** assembles the state vector (14-dim float32), converts live BGR
     camera frames to RGB, direct-resizes images to 224x224 to match the
     LeRobot converter, rearranges to (C,H,W) uint8, and sends to the server.
   - **Server** runs the full inference pipeline: prompt injection, `YamInputs` (camera name
     remapping), quantile normalization, flow matching denoising, `AbsoluteActions` (delta to
     absolute conversion), `YamOutputs` (truncate to 14 dims).
   - **Server** returns an action chunk of 60 absolute joint position targets (1 second).
   - **Client** uses `ActionChunkBroker` to execute one action per step. After `action_horizon`
     steps (default 30 = 0.5s), it re-queries the server for a fresh chunk.
3. The robot is reset to home position after each episode.

## Action Space

The client does NOT need to handle delta-to-absolute conversion. The server's output transform
chain handles everything:

1. `Unnormalize` — de-normalizes from quantile-normalized space
2. `AbsoluteActions` — adds current state back to delta joint dims (grippers already absolute)
3. `YamOutputs` — truncates from model's 32-dim output to 14 dims

The actions returned to the client are **absolute joint position targets**, ready to pass
directly to `env.step()`.

## Inference Data Flow

```
Client (conda)                          Server (uv)
─────────────────                       ────────────────
YAMBimanualEnv._get_obs()
  |
YamEnvironment.get_observation()
  - assemble state (14,) float32
  - convert live BGR frames to RGB
  - direct-resize images to 224x224
  - transpose to (C,H,W) uint8
  |
ActionChunkBroker.infer(obs)
  | (every 30 steps = 0.5s)
WebsocketClientPolicy ---websocket--->  Policy.infer(obs)
                                          - InjectDefaultPrompt
                                          - YamInputs (camera remap)
                                          - DeltaActions (no-op at inference)
                                          - Normalize (quantile)
                                          - TokenizePrompt
                                          - model.sample_actions()
                                          - Unnormalize
                                          - AbsoluteActions (delta -> absolute)
                                          - YamOutputs (truncate to 14)
                                          |
WebsocketClientPolicy <---websocket---  {"actions": (60, 14)} absolute
  |
ActionChunkBroker returns (14,) single step
  |
YamEnvironment.apply_action()
  |
YAMBimanualEnv.step(action_14dim)    <-- absolute joint positions, ready to use
```
