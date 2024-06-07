from pathlib import Path
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from pyforcesim.rl.gym_env import JSSEnv

save_path = (Path.cwd() / './test-notebooks/test_model.zip').resolve()
tensorboard_path = (Path.cwd() / './test-notebooks/tensorboard/').resolve()
# if tensorboard_path.exists():
#     shutil.rmtree(tensorboard_path)
# tensorboard_path.mkdir(parents=True, exist_ok=True)

env = JSSEnv()
print(env.observation_space.sample())
print(env.action_space.sample())
check_env(env)

env = DummyVecEnv([lambda: JSSEnv()])
# # env = make_vec_env(lambda: JSSEnv, n_envs=1)
# model = PPO('MlpPolicy', env, verbose=2, tensorboard_log=str(tensorboard_path))
# model.learn(total_timesteps=2000, progress_bar=True)
# model.save(save_path)


model = PPO.load(save_path)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)

# obs, info = env.reset()
# n_steps = 10
# for _ in range(n_steps):
#     # Random action
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
