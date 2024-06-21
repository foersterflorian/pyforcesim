from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train import FOLDER_MODEL_SAVEPOINTS, MODEL, make_env

from pyforcesim import common
from pyforcesim.rl.gym_env import JSSEnv

NUM_EPISODES: Final[int] = 10


def get_list_of_models(
    folder_models: str,
) -> list[Path]:
    pth_models = (Path.cwd() / folder_models).resolve()
    models = list(pth_models.glob(r'*.zip'))
    models.sort(reverse=True)
    return models


# model path
# load model
models = get_list_of_models(FOLDER_MODEL_SAVEPOINTS)
# pth_model = models[0]
pth_model = Path.cwd() / 'models/2024-06-21--17-30-05_pyf_sim_PPO_mask_TS-116736.zip'
model = MaskablePPO.load(pth_model)

env = JSSEnv()
# env = ActionMasker(env, action_mask_fn=mask_fn)  # type: ignore
# env = DummyVecEnv([make_env])

all_episode_rewards = []

for ep in range(NUM_EPISODES):
    # env reset
    # obs, _ = vec_env.reset()
    # terminated, truncated
    obs, _ = env.reset()
    episode_rewards = []
    episode_actions = []
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    while not (terminated or truncated):
        action_masks = get_action_masks(env)
        # print(action_masks)
        action, _states = model.predict(obs, action_masks=action_masks)  #
        action = cast(int, action.item())
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        episode_actions.append(action)

    all_episode_rewards.append(sum(episode_rewards))
    filename = f'{MODEL}_Episode_{ep}'
    _ = env.sim_env.dispatcher.draw_gantt_chart(save_html=True, filename=filename)

mean_episode_reward = np.mean(all_episode_rewards)
print(f'Mean reward: {mean_episode_reward:.2f} - Num episodes: {NUM_EPISODES}')
