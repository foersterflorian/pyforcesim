from collections.abc import Callable
from typing import Any, Final, TypeAlias, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from pyforcesim.env_builder import (
    standard_env_1_2_3_ConstIdeal,
    standard_env_1_3_7_ConstIdeal,
)
from pyforcesim.loggers import gym_env as logger
from pyforcesim.rl import agents
from pyforcesim.simulation import environment as sim

MAX_WIP_TIME: Final[int] = 300
MAX_NON_FEASIBLE: Final[int] = 20
BuilderFunc: TypeAlias = Callable[
    [bool],
    tuple[sim.SimulationEnvironment, agents.AllocationAgent],
]
BUILDER_FUNCS: Final[dict[str, BuilderFunc]] = {
    '1-2-3_ConstIdeal': standard_env_1_2_3_ConstIdeal,
    '1-3-7_ConstIdeal': standard_env_1_3_7_ConstIdeal,
}


class JSSEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render_modes': [None]}

    def __init__(
        self,
        experiment_type: str,
        seed: int = 42,
    ) -> None:
        super().__init__()
        super().reset(seed=seed)
        self.seed = seed
        # build env
        if experiment_type not in BUILDER_FUNCS:
            raise KeyError(
                (
                    f'Experiment type >>{experiment_type}<< unknown. '
                    f'Known types are: {list(BUILDER_FUNCS.keys())}'
                )
            )
        self.builder_func = BUILDER_FUNCS[experiment_type]
        self.sim_env, self.agent = self.builder_func(True)
        # action space for allocation agent is length of all associated
        # infrastructure objects
        n_machines = len(self.agent.assoc_proc_stations)
        self.action_space = gym.spaces.Discrete(n=n_machines, seed=seed)
        # Example for using image as input (channel-first; channel-last also works):
        target_system = cast(sim.ProductionArea, self.agent.assoc_system)
        min_SGI = target_system.get_min_subsystem_id()
        max_SGI = target_system.get_max_subsystem_id()
        # observation: N_machines * (res_sys_SGI, avail, WIP_time)
        machine_low = np.array([min_SGI, 0, 0])
        machine_high = np.array([max_SGI, 1, MAX_WIP_TIME])
        # observation jobs: (job_SGI, order_time)
        job_low = np.array([0, 0])
        job_high = np.array([max_SGI, 100])

        low = np.tile(machine_low, n_machines)
        high = np.tile(machine_high, n_machines)
        low = np.append(low, job_low)
        high = np.append(high, job_high)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
            seed=seed,
        )

        self.terminated: bool = False
        self.truncated: bool = False

    def step(
        self,
        action: int,
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        # process given action
        # step through sim_env till new decision should be made
        # calculate reward based on new observation
        logger.debug('Taking step in environment')
        ## ** action is provided as parameter, set action
        # should not be needed any more, empty event list is checked below
        self.agent.set_decision(action=action)

        # ** Run till next action is needed
        # execute with provided action till next decision should be made
        while not self.agent.dispatching_signal:
            # empty event list, simulation run ended
            if not self.sim_env._event_list:
                self.terminated = True
                break
            if self.agent.non_feasible_counter > MAX_NON_FEASIBLE:
                self.truncated = True
                break

            self.sim_env.step()

        # ** Calculate Reward
        # in agent class, not implemented yet
        # call from here
        reward = self.agent.calc_reward()
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in step!')

        # additional info
        info = {}

        # finalise simulation environment
        if self.terminated:
            self.sim_env.finalise()

        logger.debug(
            'Step in environment finished. Return: %s, %s, %s, %s',
            observation,
            reward,
            self.terminated,
            self.truncated,
        )

        return observation, reward, self.terminated, self.truncated, info

    def reset(
        self,
        seed: int = 42,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        logger.debug('Resetting environment')
        self.terminated = False
        self.truncated = False
        # re-init simulation environment
        self.sim_env, self.agent = self.builder_func(True)
        logger.debug('Environment re-initialised')
        # evaluate if all needed components are registered
        self.sim_env.check_integrity()
        # initialise simulation environment
        self.sim_env.initialise()

        # run till first decision should be made
        # transient condition implemented --> triggers a point in time
        # at which agent makes decisions

        # ** Run till settling process is finished
        while not self.agent.dispatching_signal:
            # empty event list, simulation run ended
            # theoretically should never be triggered unless transient condition
            # is met later than configured simulation time
            if not self.sim_env._event_list:
                self.terminated = True
                break
            self.sim_env.step()
        # feature vector already built internally when dispatching signal is set
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in reset!')
        info = {}

        logger.info('Environment reset finished')

        return observation, info

    def action_masks(self) -> npt.NDArray[np.bool_]:
        return self.agent.action_mask

    def render(self) -> None:
        _ = self.sim_env.dispatcher.draw_gantt_chart(
            use_custom_proc_station_id=True,
            dates_to_local_tz=False,
            save_html=True,
        )

    def run_without_agent(self) -> sim.SimulationEnvironment:
        env, _ = self.builder_func(False)
        env.initialise()
        env.run()
        env.finalise()

        return env
