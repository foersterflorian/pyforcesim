import re
from pathlib import Path
from typing import Final, cast

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from train import (
    BASE_FOLDER,
    EXP_TYPE,
    FOLDER_MODEL_SAVEPOINTS,
)

from pyforcesim import common
from pyforcesim.rl.gym_env import JSSEnv

NUM_EPISODES: Final[int] = 10
FILENAME_TARGET_MODEL: Final[str] = '2024-06-24--16-17-54_pyf_sim_PPO_mask_TS-2048'
model_properties_pattern = re.compile(r'(?:pyf_sim_)([\w]+)_(TS-[\d]+)$')
matches = model_properties_pattern.search(FILENAME_TARGET_MODEL)
if matches is None:
    raise ValueError(
        f'Model properties could not be extracted out of: {FILENAME_TARGET_MODEL}'
    )
ALGO_TYPE: Final[str] = matches.group(1)
TIMESTEPS: Final[str] = matches.group(2)

USE_TRAIN_CONFIG: Final[bool] = False
USER_TARGET_FOLDER: Final[str] = '2024-06-24-01__1-3-7__ConstIdeal__Util'
USER_FOLDER: Final[str] = f'results/{USER_TARGET_FOLDER}'
user_exp_type_pattern = re.compile(
    r'^([\d\-]*)(?:[_]*)([\d\-]*)(?:[_]*)([a-zA-Z]*)(?:[_]*)([a-zA-Z]*)$'
)
matches = user_exp_type_pattern.match(USER_TARGET_FOLDER)
if matches is None:
    raise ValueError(f'Experiment type could not be extracted out of: {USER_TARGET_FOLDER}')

USER_EXP_TYPE: Final[str] = f'{matches.group(2)}_{matches.group(3)}'


def get_list_of_models(
    folder_models: str,
) -> list[Path]:
    pth_models = (Path.cwd() / folder_models).resolve()
    models = list(pth_models.glob(r'*.zip'))
    models.sort(reverse=True)
    return models


def load_model() -> tuple[str, MaskablePPO]:
    # model path
    # load model
    # models = get_list_of_models(FOLDER_MODEL_SAVEPOINTS)
    # pth_model = models[0]
    # ** load model
    base_folder = USER_FOLDER
    if USE_TRAIN_CONFIG:
        base_folder = BASE_FOLDER

    pth_model = common.prepare_save_paths(
        base_folder=base_folder,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=FILENAME_TARGET_MODEL,
        suffix='zip',
    )
    # pth_model = Path.cwd() / 'models/2024-06-21--17-30-05_pyf_sim_PPO_mask_TS-116736.zip'
    model = MaskablePPO.load(pth_model)

    return base_folder, model


def test() -> str:
    # model
    base_folder, model = load_model()
    # Env
    exp_type = USER_EXP_TYPE
    if USE_TRAIN_CONFIG:
        exp_type = EXP_TYPE
    title = (
        f'Gantt Chart<br>Model(Algo: {ALGO_TYPE}, Timesteps: '
        f'{TIMESTEPS})<br>ExpType: {exp_type}'
    )

    env = JSSEnv(exp_type)

    all_episode_rewards = []

    with logging_redirect_tqdm():
        for ep in trange(NUM_EPISODES):
            # env reset
            obs, _ = env.reset()
            episode_rewards = []
            episode_actions = []
            terminated: bool = False
            truncated: bool = False
            while not (terminated or truncated):
                action_masks = get_action_masks(env)
                # print(action_masks)
                action, _states = model.predict(
                    obs,
                    action_masks=action_masks,
                    deterministic=True,
                )
                action = cast(int, action.item())
                # print(action)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)
                episode_actions.append(action)

            all_episode_rewards.append(sum(episode_rewards))
            mean_reward = np.mean(episode_rewards)
            title_reward = f'<br>Episode: {ep}, Mean Reward: {mean_reward:.4f}'
            title_chart = title + title_reward
            filename = f'{ALGO_TYPE}_{TIMESTEPS}_Episode_{ep}'
            _ = env.sim_env.dispatcher.draw_gantt_chart(
                save_html=True,
                title=title_chart,
                filename=filename,
                base_folder=base_folder,
                sort_by_proc_station=True,
            )

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f'Mean reward: {mean_episode_reward:.4f} - Num episodes: {NUM_EPISODES}')

    return base_folder


def benchmark(base_folder: str) -> None:
    exp_type = USER_EXP_TYPE
    if USE_TRAIN_CONFIG:
        exp_type = EXP_TYPE
    env = JSSEnv(exp_type)
    title_benchmark = f'Gantt Chart<br>Benchmark<br>ExpType: {exp_type}'

    sim_env = env.run_without_agent()
    _ = sim_env.dispatcher.draw_gantt_chart(
        save_html=True,
        title=title_benchmark,
        filename='Benchmark',
        base_folder=base_folder,
        sort_by_proc_station=True,
    )


def main() -> None:
    base_folder = test()
    benchmark(base_folder)


if __name__ == '__main__':
    main()
