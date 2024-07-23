import re
import warnings
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from train import (
    BASE_FOLDER,
    EXP_TYPE,
    FOLDER_MODEL_SAVEPOINTS,
    make_env,
)

from pyforcesim import common
from pyforcesim.rl.gym_env import JSSEnv

USE_TRAIN_CONFIG: Final[bool] = True
NORMALISE_OBS: Final[bool] = True
NUM_EPISODES: Final[int] = 3
# FILENAME_TARGET_MODEL: Final[str] = '2024-06-24--16-17-54_pyf_sim_PPO_mask_TS-2048'
# FILENAME_TARGET_MODEL: Final[str] = '2024-07-19--11-58-27_pyf_sim_PPO_mask_TS-102400'
FILENAME_TARGET_MODEL: Final[str] = '2024-07-22--18-37-50_pyf_sim_PPO_mask_TS-40960'

model_properties_pattern = re.compile(r'(?:pyf_sim_)([\w]+)_(TS-[\d]+)$')
matches = model_properties_pattern.search(FILENAME_TARGET_MODEL)
if matches is None:
    raise ValueError(
        f'Model properties could not be extracted out of: {FILENAME_TARGET_MODEL}'
    )
ALGO_TYPE: Final[str] = matches.group(1)
TIMESTEPS: Final[str] = matches.group(2)

# USER_TARGET_FOLDER: Final[str] = '2024-06-24-01__1-3-7__ConstIdeal__Util'
USER_TARGET_FOLDER: Final[str] = '2024-07-18-01__1-3-7__ConstIdeal__Util'
USER_FOLDER: Final[str] = f'results/{USER_TARGET_FOLDER}'
user_exp_type_pattern = re.compile(
    r'^([\d\-]*)(?:[_]*)([\d\-]*)(?:[_]*)([a-zA-Z]*)(?:[_]*)([a-zA-Z]*)$'
)
matches = user_exp_type_pattern.match(USER_TARGET_FOLDER)
if matches is None:
    raise ValueError(f'Experiment type could not be extracted out of: {USER_TARGET_FOLDER}')

USER_EXP_TYPE: Final[str] = f'{matches.group(2)}_{matches.group(3)}'

ROOT_FOLDER = USER_FOLDER
ROOT_EXP_TYPE = USER_EXP_TYPE
if USE_TRAIN_CONFIG:
    ROOT_FOLDER = BASE_FOLDER
    ROOT_EXP_TYPE = EXP_TYPE


def get_list_of_models(
    folder_models: str,
) -> list[Path]:
    pth_models = (Path.cwd() / folder_models).resolve()
    models = list(pth_models.glob(r'*.zip'))
    models.sort(reverse=True)
    return models


def load_model() -> tuple[Path, MaskablePPO]:
    # ** load model
    pth_model = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=FILENAME_TARGET_MODEL,
        suffix='zip',
    )
    vec_norm_filename = f'{FILENAME_TARGET_MODEL}_vec_norm'
    pth_vec_norm = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=vec_norm_filename,
        suffix='pkl',
    )

    model = MaskablePPO.load(pth_model)

    return pth_vec_norm, model


def test() -> None:
    # model
    pth_vec_norm, model = load_model()
    # Env
    # env = JSSEnv(ROOT_EXP_TYPE)
    env = make_env(ROOT_EXP_TYPE, None, normalise_obs=False, gantt_chart=True)
    # vec_norm: VecNormalize | None = None
    if NORMALISE_OBS:
        if not pth_vec_norm.exists():
            raise FileNotFoundError(f'VecNormalize info not found under: {pth_vec_norm}')
        env = VecNormalize.load(str(pth_vec_norm), env)
        env.training = False
        print('Normalization info loaded successfully.')

    model.set_env(env)

    mean_episode_reward, std_episode_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=NUM_EPISODES,
        deterministic=True,
        use_masking=True,
        callback=callback,
    )

    print(
        (
            f'Mean reward: {mean_episode_reward:.4f}, std: {std_episode_reward:.4f} '
            f'- Num episodes: {NUM_EPISODES}'
        )
    )


def callback(locals, *_):
    done = cast(bool, locals['done'])

    if not done:
        return

    vec_env = cast(DummyVecEnv, locals['env'])
    env = cast(JSSEnv, vec_env.envs[0])
    # env.test_on_callback()

    cum_rewards = cast(npt.NDArray[np.float32], locals['current_rewards'])
    cum_reward = cum_rewards[0]
    episode_counters = cast(npt.NDArray[np.int32], locals['episode_counts'])
    episode_num = episode_counters[0]

    title = (
        f'Gantt Chart<br>Model(Algo: {ALGO_TYPE}, Timesteps: '
        f'{TIMESTEPS})<br>ExpType: {ROOT_EXP_TYPE}'
    )
    title_reward = f'<br>Episode: {episode_num+1}, Cum Reward: {cum_reward:.4f}'
    title_chart = title + title_reward
    filename = f'{ALGO_TYPE}_{TIMESTEPS}_Episode_{episode_num}'

    gantt_chart = env.last_gantt_chart
    if gantt_chart is None:
        raise ValueError('Gantt Chart is >>None<<')

    gantt_chart.update_layout(title=title_chart, margin=dict(t=150))
    save_pth = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=None,
        filename=filename,
        suffix='html',
    )
    gantt_chart.write_html(save_pth, auto_open=False)

    # if env.last_op_db is not None:
    #     env.last_op_db.to_pickle(Path.cwd())


def test_old() -> None:
    # model
    pth_vec_norm, model = load_model()
    # Env
    title = (
        f'Gantt Chart<br>Model(Algo: {ALGO_TYPE}, Timesteps: '
        f'{TIMESTEPS})<br>ExpType: {ROOT_EXP_TYPE}'
    )

    env = JSSEnv(ROOT_EXP_TYPE)

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
                base_folder=ROOT_FOLDER,
                sort_by_proc_station=True,
            )

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f'Mean reward: {mean_episode_reward:.4f} - Num episodes: {NUM_EPISODES}')


def benchmark() -> None:
    env = JSSEnv(ROOT_EXP_TYPE)
    title_benchmark = f'Gantt Chart<br>Benchmark<br>ExpType: {ROOT_EXP_TYPE}'

    sim_env = env.run_without_agent()
    _ = sim_env.dispatcher.draw_gantt_chart(
        save_html=True,
        title=title_benchmark,
        filename='Benchmark',
        base_folder=ROOT_FOLDER,
        sort_by_proc_station=True,
    )


def main() -> None:
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message=r'^[\s]*.*to get variables from other wrappers is deprecated.*$',
    )
    test()
    # sys.exit(0)
    benchmark()


if __name__ == '__main__':
    main()
