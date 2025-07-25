from __future__ import annotations

import re
from collections.abc import Generator, Iterator, Sequence
from dataclasses import asdict
from datetime import timedelta as Timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import gymnasium as gym
import joblib
import lstm_aenc
import lstm_aenc.models
import lstm_aenc.train
import numpy as np
import numpy.typing as npt
import torch
from pandas import DataFrame

from pyforcesim import common
from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import (
    CFG_ALPHA,
    CFG_BUFFER_SIZE,
    CFG_DISPATCHER_ALLOC_RULE,
    CFG_DISPATCHER_SEQ_RULE,
    CFG_FACTOR_WIP,
    CFG_JOB_POOL_SIZE_MAX,
    CFG_JOB_POOL_SIZE_MIN,
    CFG_SIM_DUR_WEEKS,
    CFG_USE_WIP_TARGETS,
    CFG_WIP_LEVEL_CYCLES,
    CFG_WIP_RELATIVE_PLANNED,
    CFG_WIP_RELATIVE_TARGETS,
    CFG_WIP_TARGET_MAX,
    CFG_WIP_TARGET_MIN,
    CFG_WIP_TARGET_NUM_LEVELS,
    DEFAULT_SEED,
    SLACK_DEFAULT_LOWER_BOUND,
    SLACK_THRESHOLD_UPPER,
    TimeUnitsTimedelta,
)
from pyforcesim.env_builder import (
    calc_WIP_relative_targets,
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
    from sklearn.preprocessing import RobustScaler

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


def iterate_seeds(
    seeds: Sequence[int],
) -> Iterator[int]:
    max_idx = len(seeds) - 1
    idx: int = 0

    while True:
        yield seeds[idx]
        idx += 1
        if idx > max_idx:
            idx = 0


def get_builder_func_WIP_config() -> EnvBuilderAdditionalConfig:
    # calculate WIP targets or not
    WIP_relative_targets: tuple[float, ...]
    if CFG_USE_WIP_TARGETS:
        _, WIP_relative_targets = calc_WIP_relative_targets(
            CFG_WIP_TARGET_MIN,
            CFG_WIP_TARGET_MAX,
            CFG_WIP_TARGET_NUM_LEVELS,
        )
    else:
        WIP_relative_targets = CFG_WIP_RELATIVE_TARGETS

    builder_func_wip_cfg: EnvBuilderAdditionalConfig = {
        'sim_dur_weeks': CFG_SIM_DUR_WEEKS,  # 2/4/39
        'factor_WIP': CFG_FACTOR_WIP,
        'WIP_relative_target': WIP_relative_targets,
        'WIP_level_cycles': CFG_WIP_LEVEL_CYCLES,
        'WIP_relative_planned': CFG_WIP_RELATIVE_PLANNED,
        'alpha': CFG_ALPHA,
        'buffer_size': CFG_BUFFER_SIZE,
        'job_pool_size_min': CFG_JOB_POOL_SIZE_MIN,  # 1/5
        'job_pool_size_max': CFG_JOB_POOL_SIZE_MAX,
        'dispatcher_seq_rule': CFG_DISPATCHER_SEQ_RULE,
        'dispatcher_alloc_rule': CFG_DISPATCHER_ALLOC_RULE,
    }

    return builder_func_wip_cfg


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
        seeds: Sequence[int] | None = (DEFAULT_SEED,),
        sim_randomise_reset: bool = False,
        sim_check_agent_feasibility: bool = True,
        builder_func_family: BuilderFuncFamilies = BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
        seed_layout: int | None = DEFAULT_SEED,
        states_actions_path: Path | None = None,
        obs_encoder_checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        BUILDER_FUNC_WIP_CFG: Final[EnvBuilderAdditionalConfig] = (
            get_builder_func_WIP_config()
        )
        # exp type example: '1-2-3_VarIdeal_validate'
        self._verify_seed_options(seeds, sim_randomise_reset)
        self.seeds: Iterator[int] | None = None
        self.seeds_used: list[int | None] = []
        if seeds is not None:
            self.seeds = iterate_seeds(seeds)
        self.seed = self._get_seed()
        self.last_seed: int | None = None

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

        self.obs_postprocessor: ObsPostprocessor | None = None
        if obs_encoder_checkpoint is not None:
            self.obs_postprocessor = ObsPostprocessor(obs_encoder_checkpoint)

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
        elif agent_type == AgentDecisionTypes.SEQ and self.obs_postprocessor is None:
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
            machine_low = np.array([min_SGI, 0])
            machine_high = np.array([max_SGI, 1])
            # observation jobs:
            # N_queue_slots * (target_SGI, order time, lower bound slack,
            # upper bound slack, current slack)
            # lower_bound_slack_min = SLACK_DEFAULT_LOWER_BOUND / NORM_TD
            # upper_bound_slack_min = SLACK_THRESHOLD_UPPER / NORM_TD
            # if lower_bound_slack_min > 0:
            #     lower_bound_slack_min = 0
            # if upper_bound_slack_min > 0:
            #     upper_bound_slack_min = 0

            lower_bound_slack_min = -100
            upper_bound_slack_min = -100

            job_low = np.array(
                [
                    min_SGI,
                    0,
                    lower_bound_slack_min,
                    upper_bound_slack_min,
                    -1000,
                ]
            )
            job_high = np.array(
                [
                    max_SGI,
                    100,
                    100,
                    100,
                    1000,
                ]
            )

            job_low = np.tile(job_low, queue_slots)
            job_high = np.tile(job_high, queue_slots)
            low = np.append(machine_low, job_low)
            high = np.append(machine_high, job_high)

            self.observation_space = gym.spaces.Box(  # type: ignore
                low=low,
                high=high,
                dtype=np.float32,
                seed=self.seed,
            )
        elif agent_type == AgentDecisionTypes.SEQ and self.obs_postprocessor is not None:
            # TODO: add difference with observation encoder
            ...
        else:
            raise NotImplementedError('Other agent types not supported')

        # valid action sampling, works, but can not be activated since
        # this statement makes the object not pickable and therefore it can not
        # be saved with SB3's internal tools
        # self.action_space.sample = self.random_action  # type: ignore
        self.terminated: bool = False
        self.truncated: bool = False

        # saving state and actions
        self.state_saver: Generator[None, npt.NDArray, None] | None = None
        if states_actions_path is not None:
            self.state_saver = save_batches(states_actions_path, batch_size=1024)
            next(self.state_saver)
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

    @staticmethod
    def _verify_seed_options(
        seeds: Sequence[int] | None,
        sim_randomise_reset: bool,
    ) -> None:
        if sim_randomise_reset and seeds is not None:
            raise ValueError(
                '[Gym-Env] Env with randomise reset option, but a seed was provided.'
            )
        elif not sim_randomise_reset and seeds is None:
            raise ValueError(
                '[Gym-Env] Env without randomise reset option, '
                'but no seeds are present in the GymEnv.'
            )

    def _get_seed(self) -> int | None:
        seed: int | None
        if self.seeds is None:
            seed = None
        else:
            seed = next(self.seeds)

        return seed

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

        # ** state saving
        if self.state_saver is not None:
            self.state_saver.send(np.array([action], dtype=np.float32))

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

        # post-processed information
        observation: npt.NDArray[np.float32]
        if self.obs_postprocessor is None:
            assert (
                self.agent.feat_vec is not None
            ), 'tried to access non-existing feature vector'
            observation = self.agent.feat_vec
            if observation is None:
                raise ValueError('No Observation in step!')
        else:
            # TODO add observation encoding
            ...

        # additional info
        info = {}

        logger.debug(
            'Step in environment finished. Return: %s, %s, %s, %s',
            observation,
            reward,
            self.terminated,
            self.truncated,
        )
        # TODO: add check to not save obs/act if environment is terminated
        # assert self.agent.feat_vec_raw is not None
        # length_rfc = len(self.agent.feat_vec_raw)
        # if (length_rfc - 2) % 5 != 0:
        #     raise RuntimeError(
        #         f'Dubious feature vector with different size: \n{self.agent.feat_vec_raw}'
        #     )
        if self.state_saver is not None and not (self.terminated or self.truncated):
            assert (
                self.agent.feat_vec_raw is not None
            ), 'tried to process non-existent raw feature vector'
            self.state_saver.send(self.agent.feat_vec_raw)

        # TODO postprocess of observation vector with compression through encoder

        return observation, reward, self.terminated, self.truncated, info

    def reset(  # type: ignore
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        logger.debug('Resetting environment')
        self.terminated = False
        self.truncated = False
        # re-init simulation environment
        if self.sim_randomise_reset:
            self._build_env(seed=seed)
        else:
            # use provided seed, if given
            if seed is None:
                seed = self._get_seed()
            self._build_env(seed=seed)

        self.seeds_used.append(seed)
        self.last_seed = self.seed
        self.seed = seed
        super().reset(seed=self.seed)

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

        if self.state_saver is not None:
            assert (
                self.agent.feat_vec_raw is not None
            ), 'tried to process non raw feature vector'
            self.state_saver.send(self.agent.feat_vec_raw)

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


def save_batches(
    path: Path,
    batch_size: int,
    include_timestamp: bool = False,
    num_features_machine: int = 2,
    num_features_job: int = 5,
) -> Generator[None, npt.NDArray, None]:
    assert path.exists(), 'path for saving batches not existing'
    path = path.resolve()
    save_path = common.prepare_save_paths(
        str(path),
        'states-actions',
        None,
        None,
        create_folder=True,
    )
    # send observation, send action
    batch: list[npt.NDArray[np.float32]] = []
    counter: int = 1
    while True:
        obs = yield None
        action = yield None

        if len(action) == 0:
            raise RuntimeError(f'Action with length 0: \n{action}')
        elif len(action) > 1:
            raise RuntimeError(f'Action with length greater than 1: \n{action}')
        length_obs = len(obs)
        # TODO add check to not include non-feasible obs/act-pairs
        if (length_obs - num_features_machine) % num_features_job != 0:
            raise RuntimeError(
                f'Dubious feature vector with different size: shape={obs.shape}\n{obs}'
            )

        obs_act = np.concatenate((action, obs), axis=0, dtype=np.float32)
        batch.append(obs_act)

        if len(batch) == batch_size:
            filename: str = f'obs-act_batch-{counter}_batch-size-{batch_size}'
            pth_batch = common.prepare_save_paths(
                str(save_path), None, filename, '.npz', include_timestamp=include_timestamp
            )
            np.savez_compressed(pth_batch, *batch)
            counter += 1
            batch = []


class ObsPostprocessor:
    __slots__ = (
        'device',
        'path_scaler',
        'scaler',
        'path_model_checkpoint',
        'checkpoint',
        'encoder',
    )

    def __init__(
        self,
        path_scaler: Path,
        path_model_checkpoint: Path,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path_scaler = path_scaler
        logger.info('Loading scaler from file...')
        self.scaler: RobustScaler = joblib.load(self.path_scaler)
        assert isinstance(self.scaler, RobustScaler), 'scaler instance type unknown'
        logger.info('Loaded scaler successfully.')

        self.path_model_checkpoint = path_model_checkpoint
        self.checkpoint = lstm_aenc.train.load_checkpoint(path_model_checkpoint)
        self.encoder = lstm_aenc.models.Encoder.from_dump(self.checkpoint['model']['enc'])
        self.encoder.to(self.device)
        self.encoder.eval()

    def process_obs(
        self,
        raw_obs: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """encodes raw observation sequences from a buffer to a fixed-length vector

        Parameters
        ----------
        raw_obs : npt.NDArray[np.float32]
            raw observations in shape (N, 5) where N is the sequence length

        Returns
        -------
        npt.NDArray[np.float32]
            encoded observation in shape (32,)
        """

        # test case: sequences with 5 features and differing lengths
        scaled_obs = self.scaler.transform(raw_obs)  # (N, 5)
        X = torch.from_numpy(scaled_obs).unsqueeze(0).to(self.device)  # (1, N, 5)

        with torch.no_grad():
            obs_enc = cast(torch.Tensor, self.encoder(X))

        return obs_enc.squeeze().cpu().numpy()  # (32,)
