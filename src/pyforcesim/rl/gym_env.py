from datetime import timedelta as Timedelta
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from pyforcesim.constants import DEFAULT_SEED
from pyforcesim.env_builder import (
    standard_env_1_2_3_ConstIdeal,
    standard_env_1_3_7_ConstIdeal,
    standard_env_1_3_7_ConstIdeal_validate,
    standard_env_1_3_7_VarIdeal,
    standard_env_1_3_7_VarIdeal_validate,
    standard_env_1_5_5_ConstIdeal,
    standard_env_1_5_10_ConstIdeal,
    standard_env_1_5_15_ConstIdeal,
    standard_env_1_5_20_ConstIdeal,
    standard_env_1_5_30_ConstIdeal,
    standard_env_1_5_50_ConstIdeal,
    standard_env_1_5_70_ConstIdeal,
)
from pyforcesim.loggers import gym_env as logger
from pyforcesim.simulation import environment as sim
from pyforcesim.types import EnvBuilderFunc, PlotlyFigure

MAX_WIP_TIME: Final[int] = 300
MAX_NON_FEASIBLE: Final[int] = 20


BUILDER_FUNCS: Final[dict[str, EnvBuilderFunc]] = {
    '1-2-3_ConstIdeal': standard_env_1_2_3_ConstIdeal,
    '1-3-7_ConstIdeal': standard_env_1_3_7_ConstIdeal,
    '1-3-7_ConstIdeal_validate': standard_env_1_3_7_ConstIdeal_validate,
    '1-3-7_VarIdeal': standard_env_1_3_7_VarIdeal,
    '1-3-7_VarIdeal_validate': standard_env_1_3_7_VarIdeal_validate,
    '1-5-5_ConstIdeal': standard_env_1_5_5_ConstIdeal,
    '1-5-10_ConstIdeal': standard_env_1_5_10_ConstIdeal,
    '1-5-15_ConstIdeal': standard_env_1_5_15_ConstIdeal,
    '1-5-20_ConstIdeal': standard_env_1_5_20_ConstIdeal,
    '1-5-30_ConstIdeal': standard_env_1_5_30_ConstIdeal,
    '1-5-50_ConstIdeal': standard_env_1_5_50_ConstIdeal,
    '1-5-70_ConstIdeal': standard_env_1_5_70_ConstIdeal,
}


class JSSEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render_modes': [None]}

    def __init__(
        self,
        experiment_type: str,
        gantt_chart_on_termination: bool = False,
        seed: int | None = DEFAULT_SEED,
        sim_randomise_reset: bool = False,
        sim_check_agent_feasibility: bool = True,
    ) -> None:
        super().__init__()
        # super().reset(seed=seed)
        self.seed = seed
        self.sim_randomise_reset = sim_randomise_reset
        self.sim_check_agent_feasibility = sim_check_agent_feasibility
        # build env
        if experiment_type not in BUILDER_FUNCS:
            raise KeyError(
                (
                    f'Experiment type >>{experiment_type}<< unknown. '
                    f'Known types are: {tuple(BUILDER_FUNCS.keys())}'
                )
            )
        self.exp_type = experiment_type
        self.builder_func = BUILDER_FUNCS[self.exp_type]
        self.sim_env, self.agent = self.builder_func(with_agent=True, seed=self.seed)
        self.sim_env.check_agent_feasibility = self.sim_check_agent_feasibility
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
        # observation jobs: (job_SGI, order_time, slack_current)
        job_low = np.array([0, 0, -1000])
        job_high = np.array([max_SGI, 100, 1000])

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

        # process control
        self.gantt_chart_on_termination = gantt_chart_on_termination
        # external properties to handle callbacks and termination actions
        self.last_gantt_chart: PlotlyFigure | None = None
        self.last_job_db: DataFrame | None = None
        self.last_op_db: DataFrame | None = None
        self.cycle_time: Timedelta | None = None
        self.sim_utilisation: float | None = None

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

        # background: up to this point calculation of rewards always based on
        # state s_(t+1) for action a_t, but reward should be calculated
        # for state s_t --> r_t = R(s_t, a_t)
        # ** Calculate Reward
        reward = self.agent.calc_reward()

        # ** Run till next action is needed
        # execute with provided action till next decision should be made
        while not self.agent.dispatching_signal:
            # empty event list, simulation run ended
            if not self.sim_env._event_list:
                self.terminated = True
                self._on_termination(gantt_chart=self.gantt_chart_on_termination)
                break
            if self.agent.non_feasible_counter > MAX_NON_FEASIBLE:
                self.truncated = True
                break

            self.sim_env.step()

        # Calculate Reward
        # in agent class, not implemented yet
        # call from here
        # reward = self.agent.calc_reward()
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in step!')

        # additional info
        info = {}

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
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        logger.debug('Resetting environment')
        super().reset(seed=seed)
        self.terminated = False
        self.truncated = False
        # re-init simulation environment
        if self.sim_randomise_reset:
            self.sim_env, self.agent = self.builder_func(with_agent=True, seed=seed)
        else:
            self.sim_env, self.agent = self.builder_func(with_agent=True, seed=self.seed)
        self.sim_env.check_agent_feasibility = self.sim_check_agent_feasibility
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
                self._on_termination(gantt_chart=self.gantt_chart_on_termination)
                break
            self.sim_env.step()
        # feature vector already built internally when dispatching signal is set
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in reset!')
        info = {}

        logger.info('Environment reset finished.')

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
        env, _ = self.builder_func(with_agent=False, seed=self.seed)
        env.initialise()
        env.run()
        env.finalise()

        return env

    def _on_termination(
        self,
        gantt_chart: bool,
    ) -> None:
        self.sim_env.finalise()
        if gantt_chart:
            self.last_gantt_chart = self.draw_gantt_chart(sort_by_proc_station=True)
        self.last_job_db = self.sim_env.dispatcher.job_db
        self.last_op_db = self.sim_env.dispatcher.op_db
        self.cycle_time = self.sim_env.dispatcher.cycle_time
        mean_util = np.mean(self.sim_env.infstruct_mgr.final_utilisations)
        self.sim_utilisation = mean_util.item()

    def draw_gantt_chart(
        self,
        **kwargs,
    ) -> PlotlyFigure:
        """proxy to directly control the drawing and saving process of gantt charts via the
        underlying simulation environment"""
        return self.sim_env.dispatcher.draw_gantt_chart(**kwargs)

    def test_on_callback(self) -> None:
        print('CALL FROM JSSEnv')
