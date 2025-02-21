from __future__ import annotations

import re
from dataclasses import asdict
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import (
    DEFAULT_SEED,
    SLACK_THRESHOLD_LOWER,
    SLACK_THRESHOLD_UPPER,
    TimeUnitsTimedelta,
)
from pyforcesim.env_builder import (
    standard_env_single_area,
)
from pyforcesim.loggers import gym_env as logger
from pyforcesim.rl import agents
from pyforcesim.simulation import environment as sim
from pyforcesim.types import (
    AgentDecisionTypes,
    BuilderFuncFamilies,
    EnvGenerationInfo,
)

if TYPE_CHECKING:
    from pandas import Timedelta as PDTimedelta

    from pyforcesim.types import (
        EnvBuilderAdditionalConfig,
        EnvBuilderFunc,
        PlotlyFigure,
    )

MAX_WIP_TIME: Final[int] = 300
MAX_NON_FEASIBLE: Final[int] = 20
NORM_TD: Final[Timedelta] = pyf_dt.timedelta_from_val(1, time_unit=TimeUnitsTimedelta.HOURS)

BUILDER_FUNCS: Final[dict[BuilderFuncFamilies, EnvBuilderFunc]] = {
    BuilderFuncFamilies.SINGLE_PRODUCTION_AREA: standard_env_single_area,
}

WIP_TARGET_MIN: Final[float] = 0.5
WIP_TARGET_MAX: Final[float] = 5
NUM_DIFF_WIP_LEVELS: Final[int] = 5  # must be odd
assert NUM_DIFF_WIP_LEVELS % 2 != 0, 'number of WIP levels must be odd'
WIP_relative_targets: tuple[float, ...] = tuple(
    np.linspace(
        WIP_TARGET_MIN, WIP_TARGET_MAX, num=NUM_DIFF_WIP_LEVELS, dtype=np.float64
    ).tolist()
)

BUILDER_FUNC_WIP_CFG: Final[EnvBuilderAdditionalConfig] = {
    'sim_dur_weeks': 39,  # 39
    'factor_WIP': None,
    # 'WIP_relative_target': (0.5, 3, 6),
    # 'WIP_relative_target': (0.5,),
    # 'WIP_relative_target': (1.5, 0.5, 2.5, 5, 3.5),
    # 'WIP_relative_target': (3.5, 4.25, 5),
    # 'WIP_relative_target': (0.5, 1, 1.75),
    'WIP_relative_target': WIP_relative_targets,
    'WIP_level_cycles': 5,
    'WIP_relative_planned': 2.75,
    'alpha': 10,
    'buffer_size': 20,
    'job_pool_size': 1,
}


def parse_exp_type(
    exp_type: str,
) -> EnvGenerationInfo:
    pattern = re.compile(r'^[\d]+-([\d]+)-([\d]+)_([A-Z]+)_*([A-Z]*)$', re.IGNORECASE)
    matches = pattern.search(exp_type)
    if matches is None:
        raise ValueError(f'Experiment type >>{exp_type}<< could not be parsed.')

    str_num_stat_groups = matches.group(1)
    str_num_machines = matches.group(2)
    str_generation_method = matches.group(3)
    str_validation = matches.group(4)

    num_station_groups: int
    num_machines: int
    variable_source_sequence: bool
    validate: bool = False
    # convert integers
    try:
        num_station_groups = int(str_num_stat_groups)
        num_machines = int(str_num_machines)
    except ValueError:
        raise ValueError('Could not convert layout size parameters.')

    if str_generation_method == 'ConstIdeal':
        variable_source_sequence = False
    elif str_generation_method == 'VarIdeal':
        variable_source_sequence = True
    else:
        raise NotImplementedError(
            (f'Unknown source generation sequence >>{str_generation_method}<<')
        )

    if str_validation:
        validate = True

    return EnvGenerationInfo(
        num_station_groups=num_station_groups,
        num_machines=num_machines,
        variable_source_sequence=variable_source_sequence,
        validate=validate,
    )


class JSSEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render_modes': [None]}

    def __init__(
        self,
        experiment_type: str,
        agent_type: AgentDecisionTypes,
        gantt_chart_on_termination: bool = False,
        seed: int | None = DEFAULT_SEED,
        sim_randomise_reset: bool = False,
        sim_check_agent_feasibility: bool = True,
        builder_func_family: BuilderFuncFamilies = BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
        seed_layout: int | None = DEFAULT_SEED,
    ) -> None:
        super().__init__()
        # exp type example: '1-2-3_VarIdeal_validate'
        self.seed = seed
        self.seed_layout = seed_layout
        self.sim_randomise_reset = sim_randomise_reset
        self.sim_check_agent_feasibility = sim_check_agent_feasibility
        # build env
        if builder_func_family not in BUILDER_FUNCS:
            raise KeyError(
                (
                    f'Env Family >>{builder_func_family}<< unknown. '
                    f'Known types are: {tuple(BUILDER_FUNCS.keys())}'
                )
            )
        self.exp_type = parse_exp_type(experiment_type)

        self.sequencing: bool = False
        if agent_type == AgentDecisionTypes.SEQ:
            self.sequencing = True

        self.builder_func = BUILDER_FUNCS[builder_func_family]
        self.builder_kw = asdict(self.exp_type)
        self.builder_kw.update(BUILDER_FUNC_WIP_CFG)
        self.agent: agents.AllocationAgent | agents.SequencingAgent
        self._build_env(seed=self.seed)

        self.action_space: gym.spaces.Discrete
        self.observation_space: gym.spaces.Box
        if agent_type == AgentDecisionTypes.ALLOC:
            assert isinstance(
                self.agent, agents.AllocationAgent
            ), 'tried ALLOC setup for non-allocating agent'
            # action space for allocation agent is length of all associated
            # infrastructure objects
            n_machines = len(self.agent.assoc_proc_stations)
            self.action_space = gym.spaces.Discrete(n=n_machines, seed=self.seed)
            # Example for using image as input (channel-first; channel-last also works):
            target_system = cast(sim.ProductionArea, self.agent.assoc_system)
            min_SGI = target_system.get_min_subsystem_id()
            max_SGI = target_system.get_max_subsystem_id()
            # observation: N_machines * (res_sys_SGI, avail, WIP_time)
            machine_low = np.array([min_SGI, 0, 0])
            machine_high = np.array([max_SGI, 1, MAX_WIP_TIME])
            # observation jobs: (job_SGI, order_time, slack_current)
            job_low = np.array([0, 0, -1000])
            job_high = np.array([100, 100, 1000])

            low = np.tile(machine_low, n_machines)
            high = np.tile(machine_high, n_machines)
            low = np.append(low, job_low)
            high = np.append(high, job_high)

            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
                seed=self.seed,
            )
        elif agent_type == AgentDecisionTypes.SEQ:
            assert isinstance(
                self.agent, agents.SequencingAgent
            ), 'tried SEQ setup for non-sequencing agent'
            # num_actions: depending on number of queue slots
            # feature vector:
            #   - res_SGI, availability
            #   - target_SGI, order time, lower bound slack, upper bound slack,
            #       current slack
            queue_slots = self.agent.assoc_system.size
            # action space for allocation agent is number of all buffer slots
            # associated with its queue + 1 for the waiting action
            self.action_space = gym.spaces.Discrete(n=(queue_slots + 1), seed=self.seed)

            min_SGI = 0
            max_SGI = 100
            # observation: N_machines * (res_sys_SGI, avail, WIP_time)
            machine_low = np.array([min_SGI, 0])
            machine_high = np.array([max_SGI, 1])
            # observation jobs:
            # N_queue_slots * (target_SGI, order time, lower bound slack,
            # upper bound slack, current slack)
            lower_bound_slack_min = SLACK_THRESHOLD_LOWER / NORM_TD
            upper_bound_slack_min = SLACK_THRESHOLD_UPPER / NORM_TD
            if lower_bound_slack_min > 0:
                lower_bound_slack_min = 0
            if upper_bound_slack_min > 0:
                upper_bound_slack_min = 0

            job_low = np.array(
                [
                    min_SGI,
                    0,
                    lower_bound_slack_min,
                    upper_bound_slack_min,
                    -100,
                ]
            )
            job_high = np.array(
                [
                    max_SGI,
                    100,
                    lower_bound_slack_min,
                    100,
                    1000,
                ]
            )

            job_low = np.tile(job_low, queue_slots)
            job_high = np.tile(job_high, queue_slots)
            low = np.append(machine_low, job_low)
            high = np.append(machine_high, job_high)

            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
                seed=self.seed,
            )
        else:
            raise NotImplementedError('Other agent types not supported')

        # valid action sampling, works, but can not be activated since
        # this statement makes the object not pickable and therefore it can not
        # be saved with SB3's internal tools
        # self.action_space.sample = self.random_action  # type: ignore
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
        self.end_date_dev_mean: Timedelta | None = None
        self.end_date_dev_std: Timedelta | None = None
        self.policy_name: str | None = None
        # only SEQ agents
        self.waiting_chosen: int = 0
        self.jobs_total: int = 0
        self.jobs_tardy: int = 0
        self.jobs_early: int = 0
        self.jobs_punctual: int = 0

    def _build_env(
        self,
        seed: int | None,
        with_agent: bool = True,
    ) -> None:
        self.sim_env, alloc_agent, seq_agent = self.builder_func(
            sequencing=self.sequencing,
            with_agent=with_agent,
            seed=seed,
            debug=False,
            seed_layout=self.seed_layout,
            **self.builder_kw,
        )
        if alloc_agent is not None:
            self.agent = alloc_agent
        elif seq_agent is not None:
            self.agent = seq_agent
        elif with_agent:
            raise ValueError('No agent for any type available.')

        self.sim_env.check_agent_feasibility = self.sim_check_agent_feasibility

    def random_action(
        self,
        mask: npt.NDArray[np.int8] | None = None,
    ) -> int:
        return self.agent.random_action()

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
            self._build_env(seed=seed)
        else:
            self._build_env(seed=self.seed)

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
        self._build_env(seed=self.seed, with_agent=False)
        env = self.sim_env
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
        op_db = self.sim_env.dispatcher.op_db
        end_date_dev_mean = cast('PDTimedelta', op_db['ending_date_deviation'].mean())
        end_date_dev_std = cast('PDTimedelta', op_db['ending_date_deviation'].std())
        self.end_date_dev_mean = end_date_dev_mean.to_pytimedelta()
        self.end_date_dev_std = end_date_dev_std.to_pytimedelta()

        seq_policy = self.sim_env.dispatcher.seq_policy
        if seq_policy is not None:
            self.policy_name = seq_policy.name

        seq_agent: agents.SequencingAgent | None = None
        if self.sim_env.seq_agents:
            seq_agent = self.sim_env.seq_agents[0]  # only first one
        if seq_agent is not None:
            self.waiting_chosen = seq_agent.waiting_chosen
            self.jobs_total = seq_agent.jobs_total
            self.jobs_tardy = seq_agent.jobs_tardy
            self.jobs_early = seq_agent.jobs_early
            self.jobs_punctual = seq_agent.jobs_punctual

    def draw_gantt_chart(
        self,
        **kwargs,
    ) -> PlotlyFigure:
        """proxy to directly control the drawing and saving process of gantt charts via the
        underlying simulation environment"""
        return self.sim_env.dispatcher.draw_gantt_chart(**kwargs)

    def test_on_callback(self) -> None:
        print('CALL FROM JSSEnv')
