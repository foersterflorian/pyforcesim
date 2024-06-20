import shutil
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pyforcesim.rl.gym_env import JSSEnv

OVERWRITE_SAVE_FOLDER: Final[bool] = False

save_path = (Path.cwd() / './test_model.zip').resolve()
tensorboard_path = (Path.cwd() / './tensorboard/').resolve()
if OVERWRITE_SAVE_FOLDER:
    if tensorboard_path.exists():
        shutil.rmtree(tensorboard_path)
    tensorboard_path.mkdir(parents=True, exist_ok=True)
# pdm run tensorboard --logdir="./tensorboard/test_0/"


def mask_fn(env: JSSEnv) -> npt.NDArray[np.bool_]:
    return env.feasible_action_mask()


env = JSSEnv()
env = Monitor(env, filename=str(tensorboard_path), allow_early_resets=True)
env = ActionMasker(env, action_mask_fn=mask_fn)

# print(env.observation_space.sample())
# print(env.action_space.sample())
check_env(env)

vec_env = DummyVecEnv([lambda: env])
# # env = make_vec_env(lambda: JSSEnv, n_envs=1)
# model = PPO('MlpPolicy', vec_env, verbose=2, tensorboard_log=str(tensorboard_path))
model = MaskablePPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=str(tensorboard_path))
timesteps = int(2048 * 45)
model.learn(
    total_timesteps=timesteps,
    progress_bar=True,
    tb_log_name='test',
    reset_num_timesteps=True,
)
# print(f'Non-feasible counter: {env.agent.non_feasible_counter}')
model.save(save_path)


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
