# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPI is Physical Intelligence's open-source repository for Vision-Language-Action (VLA) models for robotics: π₀, π₀-FAST, and π₀.₅. Models are pre-trained on 10k+ hours of robot manipulation data and support fine-tuning via LoRA or full parameters.

The primary ML framework is **JAX** (with Flax NNX), with experimental **PyTorch** support. Python 3.11 is required.

## Common Commands

```bash
# Install dependencies (use GIT_LFS_SKIP_SMUDGE for LeRobot submodule)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Run all tests
uv run pytest --strict-markers -m "not manual"

# Run a single test file
uv run pytest src/openpi/models/pi0_test.py

# Run a single test
uv run pytest src/openpi/models/pi0_test.py::test_function_name

# Lint and format
ruff check .
ruff format .

# Install pre-commit hooks
pre-commit install

# Training (JAX)
uv run scripts/train.py --config <config_name>

# Training (PyTorch, single GPU)
uv run scripts/train_pytorch.py <config_name>

# Training (PyTorch, multi-GPU)
torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config_name>

# Compute normalization stats (required before training on new data)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Serve a policy for inference
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir>
```

## Architecture

### Core packages (`src/openpi/`)

- **models/** — JAX model implementations: `pi0.py` (flow-based), `pi0_fast.py` (autoregressive), `gemma.py`/`gemma_fast.py` (language backbone), `siglip.py` (vision encoder), `tokenizer.py` (FSQ action tokenizer), `lora.py` (LoRA adapters). Base types in `model.py`.
- **models_pytorch/** — Experimental PyTorch model ports. `transformers_replace/` patches HuggingFace transformers for precision control.
- **policies/** — Robot-specific adapters that bridge models to environments. `policy.py` defines the base Policy class. Implementations: `aloha_policy.py`, `droid_policy.py`, `libero_policy.py`.
- **training/** — Training pipeline: `config.py` (TrainConfig, DataConfig, AssetsConfig with named configs like `pi0_libero`, `pi0_fast_droid`, `debug`), `data_loader.py`, `checkpoints.py`, `optimizer.py`, `sharding.py` (JAX FSDP).
- **serving/** — `websocket_policy_server.py` for remote inference over WebSocket.
- **transforms.py** — Data transformation pipeline for preprocessing observations/actions.
- **shared/** — Utilities: array typing, normalization stats, image tools, checkpoint downloading.

### Workspace packages

- **packages/openpi-client/** — Lightweight WebSocket client for policy inference. Minimal dependencies (msgpack, numpy, websockets), Python 3.7+ compatible. Published as `openpi-client`.

### Key scripts (`scripts/`)

- `train.py` / `train_pytorch.py` — Training entrypoints (JAX / PyTorch)
- `serve_policy.py` — Policy server
- `compute_norm_stats.py` — Dataset normalization statistics

### Data flow

Training configs in `training/config.py` define dataset sources, model architecture, and training hyperparameters. Data is loaded via LeRobot datasets, preprocessed through `transforms.py`, and fed to models. Policies wrap trained models for robot deployment, served via WebSocket.

## Testing

- Tests live alongside source files as `*_test.py`
- `conftest.py` auto-detects GPU availability and falls back to JAX CPU backend
- The `manual` pytest marker is used for tests requiring special setup (excluded from CI)
- CI runs on a custom `openpi-verylarge` runner with GPU access

## Linting

- **Ruff** with line-length 120, target Python 3.11
- Imports: force single-line, sorted within sections
- `F722` ignored (conflicts with JAX array typing annotations)
- `T201` ignored (print statements are allowed)
- Excludes: `docker/`, `third_party/`, `src/openpi/models_pytorch/transformers_replace/`
