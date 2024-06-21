import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pyforcesim import common
from pyforcesim import datetime as pyf_dt
from pyforcesim.rl.gym_env import JSSEnv

OVERWRITE_FOLDERS: Final[bool] = True
TIMESTEPS_PER_ITER: Final[int] = int(2048 * 3)
ITERATIONS: Final[int] = 20
ITERATIONS_TILL_SAVE: Final[int] = 1
MODEL: Final[str] = 'PPO_mask'
FOLDER_TB: Final[str] = 'tensorboard'
FOLDER_MODEL_SAVEPOINTS: Final[str] = 'models'


def prepare_tb_path() -> Path:
    tensorboard_path = common.prepare_save_paths(FOLDER_TB, None, None)
    if OVERWRITE_FOLDERS:
        if tensorboard_path.exists():
            shutil.rmtree(tensorboard_path)
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    # pdm run tensorboard --logdir="./tensorboard/PPO_mask_0/"
    return tensorboard_path


def prepare_model_path() -> None:
    model_save_pth = common.prepare_save_paths(FOLDER_MODEL_SAVEPOINTS, None, None)
    if OVERWRITE_FOLDERS:
        if model_save_pth.exists():
            shutil.rmtree(model_save_pth)
    model_save_pth.mkdir(parents=True, exist_ok=True)


def get_save_path_model(
    num_timesteps: int,
    base_name: str = 'pyfsim_model',
) -> Path:
    filename: str = f'{base_name}_TS-{num_timesteps}'
    model_save_pth = common.prepare_save_paths(
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=filename,
        suffix='zip',
        include_timestamp=True,
    )

    return model_save_pth


def make_env(tensorboard_path: Path) -> Any:
    env = JSSEnv()
    check_env(env, warn=True)
    env = Monitor(env, filename=str(tensorboard_path), allow_early_resets=True)
    return env


def train() -> None:
    tensorboard_path = prepare_tb_path()
    prepare_model_path()
    print(tensorboard_path)

    # def mask_fn(env: JSSEnv) -> npt.NDArray[np.bool_]:
    #     return env.feasible_action_mask()

    # env = JSSEnv()
    # env = Monitor(env, filename=str(tensorboard_path), allow_early_resets=True)
    # env = ActionMasker(env, action_mask_fn=mask_fn)  # type: ignore

    # print(env.observation_space.sample())
    # print(env.action_space.sample())

    # vec_env = DummyVecEnv([make_env])
    env = make_env(tensorboard_path)

    # env = make_vec_env(lambda: JSSEnv, n_envs=1)
    # model = PPO('MlpPolicy', vec_env, verbose=2, tensorboard_log=str(tensorboard_path))
    model = MaskablePPO('MlpPolicy', env, verbose=1, tensorboard_log=str(tensorboard_path))

    for it in range(1, (ITERATIONS + 1)):
        model.learn(
            total_timesteps=TIMESTEPS_PER_ITER,
            progress_bar=True,
            tb_log_name=f'{MODEL}',
            reset_num_timesteps=False,
        )
        num_timesteps: int = TIMESTEPS_PER_ITER * it
        save_path_model = get_save_path_model(
            num_timesteps=num_timesteps, base_name=f'pyf_sim_{MODEL}'
        )
        if it % ITERATIONS_TILL_SAVE == 0:
            model.save(save_path_model)

    print('--------------------------')
    print('Training finished.')


# model = PPO.load(save_path)
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)

# obs, info = env.reset()
# n_steps = 10
# for _ in range(n_steps):
#     # Random action
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()


def main() -> None:
    train()


if __name__ == '__main__':
    main()
