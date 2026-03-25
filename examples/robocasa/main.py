import collections
import dataclasses
import logging
import math
import pathlib
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os
import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
import gymnasium as gym
from robocasa.utils.env_utils import convert_action

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    task_set: list = None
    # Direct env name (bypasses task_set registry lookup)
    env_name: str | None = None
    # Scene restriction: list of (layout_id, style_id) tuples
    layout_and_style_ids: list[tuple[int, int]] | None = None

    # Eval initialization mode
    eval_init_mode: str | None = None  # "exact_state_replay" | "fixture_pair_fresh_placement" | "fixture_pair_object_pool"
    dataset_path: str | None = None  # required for eval modes (to load states.npz / ep_meta)
    eval_pool_episode_ids: list[int] | None = None
    eval_pool_fixture_refs: dict[str, str] | None = None
    eval_pool_object_categories: list[str] | None = None
    eval_keep_robot_pose: bool = False
    # Additive uniform noise on robot base pos (xy) and yaw; 0 = exact
    eval_robot_pose_noise: float = 0.0
    # Additive uniform noise on main object XY position; 0 = exact
    eval_object_pose_noise: float = 0.0
    # Additive uniform noise on main object yaw (radians); 0 = exact
    eval_object_ori_noise: float = 0.0
    # If True, disable the gripper-must-be-far-from-object success requirement
    skip_gripper_far_check: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    if args.skip_gripper_far_check:
        import robocasa.utils.object_utils as OU
        OU.gripper_obj_far = lambda *args, **kwargs: True
        logging.info("Patched gripper_obj_far to always return True")

    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    host = args.host
    port = args.port

    if args.env_name is not None:
        all_env_names = [args.env_name]
    else:
        all_env_names = []
        for task in args.task_set:
            env_names = TASK_SET_REGISTRY[task]
            all_env_names.extend(env_names)

    for env_name in all_env_names:
        eval_env(env_name, split, log_dir, num_trials, resize_size, replan_steps, host, port, args.seed,
                 layout_and_style_ids=args.layout_and_style_ids,
                 eval_init_mode=args.eval_init_mode,
                 dataset_path=args.dataset_path,
                 eval_pool_episode_ids=args.eval_pool_episode_ids,
                 eval_pool_fixture_refs=args.eval_pool_fixture_refs,
                 eval_pool_object_categories=args.eval_pool_object_categories,
                 eval_keep_robot_pose=args.eval_keep_robot_pose,
                 eval_robot_pose_noise=args.eval_robot_pose_noise,
                 eval_object_pose_noise=args.eval_object_pose_noise,
                 eval_object_ori_noise=args.eval_object_ori_noise)

def eval_env(env_name, split, log_dir, num_trials, resize_size, replan_steps, host, port, seed,
             layout_and_style_ids=None, eval_init_mode=None, dataset_path=None,
             eval_pool_episode_ids=None, eval_pool_fixture_refs=None,
             eval_pool_object_categories=None, eval_keep_robot_pose=False,
             eval_robot_pose_noise=0.0, eval_object_pose_noise=0.0,
             eval_object_ori_noise=0.0):
    # set args based on task
    assert split in ["pretrain", "target"]
    task_horizon = get_task_horizon(env_name)
    # set dataset path and horizon
    horizon = int(task_horizon * 1.5) # the policy moves slow so give the policy extra time

    now = datetime.now()
    now_formatted = now.strftime("%Y-%m-%d-%H-%M")
    log_path = f"{log_dir}/evals/{split}/{env_name}/{now_formatted}"

    for root, dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)


    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task
    env_kwargs = {"seed": seed}
    if layout_and_style_ids is not None:
        env_kwargs["split"] = None
        env_kwargs["layout_and_style_ids"] = layout_and_style_ids
    else:
        env_kwargs["split"] = split

    # Construct eval reset controller if eval_init_mode is configured
    if eval_init_mode is not None:
        # Import from dsrl_pi0 project root (cwd when this script is run)
        import sys
        dsrl_root = os.environ.get("DSRL_PI0_ROOT", os.getcwd())
        if dsrl_root not in sys.path:
            sys.path.insert(0, dsrl_root)
        from examples.robocasa_eval_reset import RoboCasaEvalResetController
        if dataset_path is None:
            raise ValueError("--dataset-path is required when using --eval-init-mode")
        eval_controller = RoboCasaEvalResetController(
            dataset_path=pathlib.Path(dataset_path),
            eval_init_mode=eval_init_mode,
            layout_and_style_ids=layout_and_style_ids,
            eval_pool_episode_ids=eval_pool_episode_ids,
            eval_pool_fixture_refs=eval_pool_fixture_refs,
            eval_pool_object_categories=eval_pool_object_categories,
            keep_robot_pose=eval_keep_robot_pose,
            robot_pose_noise=eval_robot_pose_noise,
            object_pose_noise=eval_object_pose_noise,
            object_ori_noise=eval_object_ori_noise,
            rng_seed=seed,
        )
        env_kwargs["eval_reset_controller"] = eval_controller

    env = gym.make(f"robocasa/{env_name}", **env_kwargs)
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials)):

        # Reset environment
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )

            # Save preprocessed image for replay video
            # replay_images.append(img)

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ), axis=0
                )
                # state = np.ascontiguousarray(state)
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: replan_steps])

            action = action_plan.popleft()
            action = convert_action(action)
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"] # for robocasa, usuccess entry in info
            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            replay_img = image_tools.convert_to_uint8(
                replay_img
            )
            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        # Copy oracle (recorded demo) video from dataset if available
        if eval_init_mode is not None and "eval_reset_controller" in env_kwargs:
            ctrl = env_kwargs["eval_reset_controller"]
            reset_info = ctrl.last_reset_info
            if reset_info is not None:
                import shutil
                ep_id = reset_info["episode_id"]
                ds_path = pathlib.Path(dataset_path)
                oracle_src = ds_path / "videos" / "chunk-000" / "observation.images.robot0_agentview_left" / f"episode_{ep_id:06d}.mp4"
                if oracle_src.exists():
                    oracle_dst = pathlib.Path(log_path) / f"oracle_{episode_idx}_ep{ep_id}.mp4"
                    shutil.copy2(oracle_src, oracle_dst)

        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        stats = {
            "num_episodes": total_episodes,
            "success_rate": float(total_successes) / float(total_episodes),
        }
        json.dump(stats, f, indent=4)

    # close and delete the env
    env.env.close()
    del env.env
    del env




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)