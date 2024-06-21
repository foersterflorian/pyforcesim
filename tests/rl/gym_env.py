import gymnasium as gym
import numpy as np

from pyforcesim.rl.gym_env import JSSEnv


def test_jsse_env():
    env = JSSEnv()

    for _ in range(10):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()


def main():
    from gymnasium.spaces import Box

    # TODO use number of SGI to define upper bound
    # TODO use reasonable value for WIP and order time
    # observation: N_machines * (res_sys_SGI, avail, WIP_time)
    machine_low = np.array([0, 0, 0])
    machine_high = np.array([1000, 1, 1000])
    NUM_MACHINES = 3
    # observation jobs: (job_SGI, order_time)
    job_low = np.array([0, 0])
    job_high = np.array([1000, 1000])

    low = np.tile(machine_low, NUM_MACHINES)
    high = np.tile(machine_high, NUM_MACHINES)
    low = np.append(low, job_low)
    high = np.append(high, job_high)
    # print(low, high)

    obs = gym.spaces.Box(
        low=low,
        high=high,
        dtype=np.float32,
    )
    print(obs)
    print(obs.sample())


if __name__ == '__main__':
    # test_jsse_env()
    main()
