from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Final, Generic, TypeVar, cast
from typing_extensions import override

import numpy as np
import numpy.typing as npt
from numpy.random._generator import Generator as NPRandomGenerator

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.constants import (
    EPSILON,
    HELPER_STATES,
    ROUNDING_PRECISION,
    SEQUENCING_WAITING_TIME,
    UTIL_PROPERTIES,
    TimeUnitsTimedelta,
)
from pyforcesim.types import (
    AgentTasks,
    LoadDistribution,
    StateTimes,
    SystemID,
)

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        InfrastructureObject,
        Job,
        LogicalQueue,
        Operation,
        ProcessingStation,
        ProductionArea,
        SimulationEnvironment,
        StationGroup,
        System,
    )

S = TypeVar('S', bound='System')


class Agent(Generic[S], ABC):
    def __init__(
        self,
        assoc_system: S,
        agent_task: AgentTasks,
        seed: int | None = None,
    ) -> None:
        self._agent_task = agent_task
        self._assoc_system = assoc_system
        self._env = self._assoc_system.register_agent(agent=self, agent_task=self._agent_task)

        if seed is None and self.env.seed is not None:
            seed = self.env.seed
        self._seed = seed
        self._rng = np.random.default_rng(seed=self.seed)
        self.NORM_TD: Final[Timedelta] = pyf_dt.timedelta_from_val(
            1.0, TimeUnitsTimedelta.HOURS
        )

        self._dispatching_signal: bool = False
        self.num_decisions: int = 0
        self.cum_reward: float = 0.0
        # RL related properties
        self.feat_vec: npt.NDArray[np.float32] | None = None
        # action chosen by RL agent
        self._action: int | None = None
        self._action_mask: npt.NDArray[np.bool_] = np.empty((1,), dtype=np.bool_)
        self._action_feasible: bool = False
        self.past_action_feasible: bool = False
        # count number of consecutive non-feasible actions
        self.non_feasible_counter: int = 0

    def __key(self) -> tuple[str, SystemID]:
        return (self._agent_task, self._assoc_system.system_id)

    def __hash__(self) -> int:
        return hash(self.__key())

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}(type={self._agent_task}, '
            f'Assoc_Syst_ID={self._assoc_system.system_id})'
        )

    @property
    def assoc_system(self) -> S:
        return self._assoc_system

    @property
    def agent_task(self) -> str:
        return self._agent_task

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def rng(self) -> NPRandomGenerator:
        return self._rng

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def dispatching_signal(self) -> bool:
        return self._dispatching_signal

    @property
    def action(self) -> int | None:
        return self._action

    @property
    def action_mask(self) -> npt.NDArray[np.bool_]:
        return self._action_mask

    @action_mask.setter
    def action_mask(
        self,
        val: npt.NDArray[np.bool_],
    ) -> None:
        self._action_mask = val

    @property
    def action_feasible(self) -> bool:
        return self._action_feasible

    @action_feasible.setter
    def action_feasible(
        self,
        feasible: bool,
    ) -> None:
        if feasible:
            self.non_feasible_counter = 0
        else:
            self.non_feasible_counter += 1
        self.past_action_feasible = self._action_feasible
        self._action_feasible = feasible

    def random_action(self) -> int:
        """
        Generate random action based on associated objects
        """
        feasible_actions = np.flatnonzero(self.action_mask)
        return self.rng.choice(feasible_actions)

    def set_dispatching_signal(
        self,
        reset: bool = False,
    ) -> None:
        # check flag and determine value
        if not reset:
            # check if already set
            if not self._dispatching_signal:
                self._dispatching_signal = True
            else:
                raise RuntimeError(f'Dispatching signal for >>{self}<< was already set.')
        # reset
        else:
            # check if already not set
            if self._dispatching_signal:
                self._dispatching_signal = False
            else:
                raise RuntimeError(f'Dispatching signal for >>{self}<< was already reset.')

        loggers.agents.debug(
            'Dispatching signal for >>%s<< was set to >>%s<<.', self, self.dispatching_signal
        )

    @abstractmethod
    def activate(self) -> None: ...

    @abstractmethod
    def request_decision(self) -> None: ...

    @abstractmethod
    def set_decision(self) -> None: ...

    @abstractmethod
    def build_feat_vec(self) -> npt.NDArray[np.float32]: ...

    @abstractmethod
    def calc_reward(self) -> float: ...


class AllocationAgent(Agent['ProductionArea']):
    def __init__(
        self,
        assoc_system: ProductionArea,
        seed: int | None = None,
    ) -> None:
        # init base class
        super().__init__(assoc_system=assoc_system, agent_task='ALLOC', seed=seed)

        # get associated systems
        self._assoc_proc_stations = self.assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )
        # default action mask: allow all actions
        self._action_mask = np.tile([True], len(self._assoc_proc_stations))

        # job-related properties
        self._current_job: Job | None = None
        self._last_job: Job | None = None
        self._current_op: Operation | None = None
        self._last_op: Operation | None = None

        # states
        self.state_times_last: dict[SystemID, StateTimes] = {}
        self.state_times_current: dict[SystemID, StateTimes] = {}
        self.utilisations: dict[SystemID, float] = {}
        self.last_station_group: StationGroup | None = None

        # execution control
        # [ACTIONS]
        self._chosen_station: ProcessingStation | None = None

    @property
    def current_job(self) -> Job | None:
        return self._current_job

    @property
    def last_job(self) -> Job | None:
        return self._last_job

    @property
    def current_op(self) -> Operation | None:
        return self._current_op

    @property
    def last_op(self) -> Operation | None:
        return self._last_op

    @property
    def chosen_station(self) -> ProcessingStation | None:
        return self._chosen_station

    @property
    def assoc_proc_stations(self) -> tuple[ProcessingStation, ...]:
        return self._assoc_proc_stations

    @Agent.action_mask.setter
    @override
    def action_mask(
        self,
        val: npt.NDArray[np.bool_],
    ) -> None:
        if len(val) != len(self._action_mask):
            raise ValueError('Action mask length does not match number of resources.')
        self._action_mask = val

    def update_assoc_proc_stations(self) -> None:
        # get associated systems
        self._assoc_proc_stations = self.assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )

    @override
    def activate(self) -> None:
        for station in self.assoc_proc_stations:
            self.state_times_last[station.system_id] = station.stat_monitor.state_times.copy()
            self.state_times_current[station.system_id] = (
                station.stat_monitor.state_times.copy()
            )
            self.utilisations[station.system_id] = 0.0

    @staticmethod
    def calc_util_state_time_diff(
        state_times_last: StateTimes,
        state_times_current: StateTimes,
    ) -> float:
        state_times_diff: StateTimes = {}
        time_total = Timedelta()
        time_non_helpers = Timedelta()
        time_utilisation = Timedelta()
        utilisation: float = 0.0

        for state, time_current in state_times_current.items():
            time_diff = time_current - state_times_last[state]
            state_times_diff[state] = time_diff
            time_total += time_diff
            if state not in HELPER_STATES:
                time_non_helpers += time_diff
            if state in UTIL_PROPERTIES:
                time_utilisation += time_diff

        if time_non_helpers.total_seconds() > 0:
            utilisation = round(time_utilisation / time_non_helpers, 4)

        return utilisation

    def set_state_times_last(
        self,
        stations: Iterable[ProcessingStation],
        set_current_as_last: bool = False,
    ) -> None:
        for station in stations:
            sys_id = station.system_id
            if set_current_as_last:
                self.state_times_last[sys_id] = station.stat_monitor.state_times.copy()
            else:
                self.state_times_last[sys_id] = self.state_times_current[sys_id].copy()
            loggers.agents.debug(
                ('[AllocationAgent] State times for SystemID %d is - \n\tlast: %s'),
                sys_id,
                self.state_times_last[sys_id],
            )

    def set_state_times_current(
        self,
        stations: Iterable[ProcessingStation],
    ) -> None:
        for station in stations:
            sys_id = station.system_id
            self.state_times_current[sys_id] = station.stat_monitor.state_times.copy()
            loggers.agents.debug(
                ('[AllocationAgent] State times for SystemID %d is - \n\tcurrent: %s'),
                sys_id,
                self.state_times_current[sys_id],
            )

    def calc_utilisation(
        self,
    ) -> None:
        for station in self.assoc_proc_stations:
            sys_id = station.system_id
            utilisation = self.calc_util_state_time_diff(
                state_times_last=self.state_times_last[sys_id],
                state_times_current=self.state_times_current[sys_id],
            )
            self.utilisations[sys_id] = utilisation
            loggers.agents.debug(
                '[AllocationAgent] Utilisation for SystemID %d is %.4f', sys_id, utilisation
            )

        loggers.agents.debug(
            '[AllocationAgent] Utilisation dict for agent >>%s<<: %s', self, self.utilisations
        )

    @override
    def request_decision(
        self,
        job: Job,
        op: Operation,
    ) -> None:
        # indicator that request is being made
        self.set_dispatching_signal(reset=False)

        # remember relevant jobs
        if self.current_job is None or job != self.current_job:
            self._last_job = self._current_job
            self._current_job = job
        if self.current_op is None or op != self._current_op:
            self._last_op = self._current_op
            self._current_op = op

        loggers.agents.debug(
            '[AllocationAgent] Current OP for agent >>%s<<: %s', self, self._current_op
        )

        # TODO change: only update relevant stations of station group
        assert self.current_op is not None, 'Current OP must not be >>None<<'
        relevant_stat_group = self.current_op.target_station_group
        assert relevant_stat_group is not None
        relevant_stations = relevant_stat_group.assoc_proc_stations
        current_as_last: bool = False
        if self.last_station_group is None or self.last_station_group != relevant_stat_group:
            current_as_last = True
            self.last_station_group = relevant_stat_group
        self.set_state_times_last(
            stations=relevant_stations,
            set_current_as_last=current_as_last,
        )
        # self.calc_utilisation(stations=relevant_stations)

        # build feature vector
        self.feat_vec = self.build_feat_vec(job=job)

        loggers.agents.debug(
            '[REQUEST Agent %s]: built FeatVec at %s', self, self.env.t_as_dt()
        )
        loggers.agents.debug('[Agent %s]: Feature Vector: %s', self, self.feat_vec)

    # @override
    def request_decision_backup(
        self,
        job: Job,
        op: Operation,
    ) -> None:
        # indicator that request is being made
        self.set_dispatching_signal(reset=False)

        # remember relevant jobs
        if self.current_job is None or job != self.current_job:
            self._last_job = self._current_job
            self._current_job = job
        if self.current_op is None or op != self._current_op:
            self._last_op = self._current_op
            self._current_op = op

        loggers.agents.debug(
            '[AllocationAgent] Current OP for agent >>%s<<: %s', self, self._current_op
        )
        # TODO change: only update relevant stations of station group
        for station in self.assoc_proc_stations:
            sys_id = station.system_id
            self.state_times_last[sys_id] = self.state_times_current[sys_id].copy()
            self.state_times_current[sys_id] = station.stat_monitor.state_times.copy()
            # calculate difference between last and current for each station
            # calculate utilisation (difference based) for each station
            loggers.agents.debug(
                (
                    '[AllocationAgent] State times for SystemID %d is - \n\t  '
                    ' last: %s, \n\tcurrent: %s'
                ),
                sys_id,
                self.state_times_last[sys_id],
                self.state_times_current[sys_id],
            )
            utilisation = self.calc_util_state_time_diff(
                state_times_last=self.state_times_last[sys_id],
                state_times_current=self.state_times_current[sys_id],
            )
            self.utilisations[sys_id] = utilisation
            loggers.agents.debug(
                '[AllocationAgent] Utilisation for SystemID %d is %.4f', sys_id, utilisation
            )

        loggers.agents.debug(
            '[AllocationAgent] Utilisation dict for agent >>%s<<: %s', self, self.utilisations
        )

        # build feature vector
        self.feat_vec = self.build_feat_vec(job=job)

        loggers.agents.debug(
            '[REQUEST Agent %s]: built FeatVec at %s', self, self.env.t_as_dt()
        )
        loggers.agents.debug('[Agent %s]: Feature Vector: %s', self, self.feat_vec)

    @override
    def set_decision(
        self,
        action: int,
    ) -> None:
        self._action = action
        self._chosen_station = self.assoc_proc_stations[action]
        self.num_decisions += 1
        # indicator that request was processed, reset dispatching signal
        self.set_dispatching_signal(reset=True)

        loggers.agents.debug('[DECISION SET Agent %s]: Set to >>%d<<', self, self._action)

    @override
    def build_feat_vec(
        self,
        job: Job,
    ) -> npt.NDArray[np.float32]:
        action_mask: list[bool] = []
        station_feasible: bool
        # job
        # needed properties: target station group ID, order time
        if job.current_order_time is None:
            raise ValueError(f'Current order time of job {job} >>None<<')
        order_time = job.current_order_time / self.NORM_TD
        slack = job.stat_monitor.slack_hours
        # current op: obtain StationGroupID
        current_op = job.current_op
        if current_op is not None:
            job_SGI = current_op.target_station_group_identifier
        else:
            raise ValueError(
                'Tried to build feature vector for job without current operation.'
            )
        if job_SGI is None:
            raise ValueError('Station Group ID of current operation is None.')
        job_info = (job_SGI, order_time, slack)
        job_info_arr = np.array(job_info, dtype=np.float32)
        # resources
        # needed properties
        # station group, availability, WIP_time
        for i, res in enumerate(self.assoc_proc_stations):
            # T1 build feature vector for one machine
            monitor = res.stat_monitor
            # station group identifier should be the system's one
            # because custom IDs can be non-numeric which is bad for an agent
            # use only first identifier although multiple values are possible
            supersystem = cast('StationGroup', res.supersystems_as_list()[0])
            res_SGI = supersystem.system_id
            # feasibility check
            if res_SGI == job_SGI:
                station_feasible = True
            else:
                station_feasible = False
            action_mask.append(station_feasible)

            # availability: boolean to integer
            avail = int(monitor.is_available)
            # WIP_time in hours
            # WIP_time = monitor.WIP_load_time / Timedelta(hours=1)
            WIP_time_remain = monitor.WIP_load_time_remaining / Timedelta(hours=1)
            # tuple: (System SGI of resource obj, availability status,
            # WIP calculated in time units)
            # res_info = (res_SGI, avail, WIP_time)
            res_info = (res_SGI, avail, WIP_time_remain)
            res_info_arr_current = np.array(res_info, dtype=np.float32)

            if i == 0:
                res_info_arr_all = res_info_arr_current
            else:
                res_info_arr_all = np.concatenate((res_info_arr_all, res_info_arr_current))

        # concat job information
        res_info_arr_all = np.concatenate((res_info_arr_all, job_info_arr), dtype=np.float32)
        # action mask
        self.action_mask = np.array(action_mask, dtype=np.bool_)

        return res_info_arr_all

    @staticmethod
    def load_distribution_diff(
        ideal: LoadDistribution,
        actual: LoadDistribution,
        availabilities: tuple[bool, ...] | None = None,
    ) -> float:
        """load distribution score between [0;2] where 0 is best, 2 is worst"""
        load_ideal = np.array(tuple(ideal.values()), dtype=np.float64)
        load_actual = np.array(tuple(actual.values()), dtype=np.float64)
        diff = cast(npt.NDArray[np.float64], np.abs(load_ideal - load_actual))

        mask: npt.NDArray[np.bool_]
        if availabilities is not None:
            arr_availabilities = cast(npt.NDArray[np.bool_], np.array(availabilities))
            mask = (np.logical_not(arr_availabilities) * load_actual) > 0
        else:
            mask = np.ones(shape=len(load_actual), dtype=np.bool_)

        score = cast(float, np.sum(diff * mask))
        loggers.agents.debug(
            '[Agent Reward - Load] ideal: %s, actual: %s, score: %.6f',
            load_ideal,
            load_actual,
            score,
        )

        return score

    def reward_load_balancing(
        self,
        target_station_group: StationGroup,
        chosen_station: ProcessingStation,
        op_reward: Operation,
        enable_masking: bool,
    ) -> float:
        load_distribution_ideal = target_station_group.load_distribution_ideal
        load_distribution_action = target_station_group.workload_distribution_future(
            target_system_id=chosen_station.system_id,
            new_load=op_reward.order_time,
        )
        availabilities: tuple[bool, ...] | None = None
        if enable_masking:
            availabilities = target_station_group.get_proc_station_availability()

        score = self.load_distribution_diff(
            ideal=load_distribution_ideal,
            actual=load_distribution_action,
            availabilities=availabilities,
        )
        # load distribution score between [0;2]
        # normalise in range [-1;1]
        # reward = (-1) * score
        reward = (score - 1) * (-1)

        return reward

    @staticmethod
    def _reward_sigmoid(
        x: float,
        beta: float = 1.0,
    ) -> float:
        return x / (abs(x) + beta)

    @staticmethod
    def _reward_gaussian(
        x: float,
        b: float = 0.0,
        c: float = 1.0,
    ) -> float:
        return 2 * np.exp(-np.power((x - b), 2) / (2 * np.power(c, 2))) - 1

    def reward_slack_delta_current(
        self,
        chosen_station: ProcessingStation,
        c_gaussian: float = 2.5,
    ) -> float:
        # current slack
        # job = op_reward.job
        # slack_current = job.stat_monitor.slack_hours
        # calc average slack
        # workload_total = target_station_group.workload_current(total=True)
        # workload_capacity_total = target_station_group.processing_capacities(total=True)
        # t_till_work_done = workload_total / workload_capacity_total
        # slack_average = current_slack - t_till_work_done
        # calc actual slack (only applicable if FIFO is chosen -> deterministic behaviour)
        workload_station = chosen_station.stat_monitor.WIP_load_time_remaining
        workload_capacity_station = chosen_station.processing_capacity
        t_till_work_done_station = workload_station / workload_capacity_station
        slack_delta = -t_till_work_done_station

        reward = self._reward_gaussian(slack_delta, c=c_gaussian)

        return reward

    def reward_slack_estimate(
        self,
        target_station_group: StationGroup,
        chosen_station: ProcessingStation,
        op_reward: Operation,
        beta_sigmoid: float = 1.0,
    ) -> float:
        # current slack
        job = op_reward.job
        current_slack = job.stat_monitor.slack_hours
        # calc average slack
        workload_total = target_station_group.workload_current(total=True)
        workload_capacity_total = target_station_group.processing_capacities(total=True)
        t_till_work_done = workload_total / workload_capacity_total
        slack_average = current_slack - t_till_work_done
        # calc actual slack (only applicable if FIFO is chosen -> deterministic behaviour)
        workload_station = chosen_station.stat_monitor.WIP_load_time_remaining
        workload_capacity_station = chosen_station.processing_capacity
        t_till_work_done_station = workload_station / workload_capacity_station
        slack_actual = current_slack - t_till_work_done_station
        # calc reward as difference between average and actual slack
        slack_delta = slack_actual - slack_average
        # use sigmoid curvature: increased slack positive, decreased negative
        reward = self._reward_sigmoid(slack_delta, beta=beta_sigmoid)

        return reward

    @override
    def calc_reward(self) -> float:
        op_rew = self.current_op
        if op_rew is None:
            raise ValueError('Tried to calculate reward based on a non-existent operation.')
        # obtain target station (set during action setting)
        chosen_station = self.chosen_station
        if chosen_station is None:
            raise ValueError(
                "No station was chosen. Maybe the agent's action was not properly set."
            )
        # obtain target SG from respective operation
        target_station_group = op_rew.target_station_group
        if target_station_group is None:
            raise ValueError(f'Operation >>{op_rew}<< has no associated station group.')

        loggers.agents.debug('[Agent Reward] Target station group: %s', target_station_group)
        loggers.agents.debug('[Agent Reward] Chosen station: %s', chosen_station)
        # perform feasibility check + set flag
        self.action_feasible = self.env.check_feasible_agent_choice(
            target_station=chosen_station, op=op_rew
        )

        reward: float
        if self.action_feasible:
            # load_distribution_current = target_station.workload_distribution_current()
            # reward = self.reward_load_balancing(
            #     target_station_group=target_station_group,
            #     chosen_station=chosen_station,
            #     op_reward=op_rew,
            #     enable_masking=False,
            # )
            # reward = self.reward_slack_estimate(
            #     target_station_group=target_station_group,
            #     chosen_station=chosen_station,
            #     op_reward=op_rew,
            # )
            reward = self.reward_slack_delta_current(
                chosen_station=chosen_station,
            )

        else:
            # non-feasible actions
            reward = -100.0

        loggers.agents.debug('[AllocationAgent] reward: %.4f', reward)
        self.cum_reward += reward
        loggers.agents.debug('[Agent Reward] Num Decisions: %d', self.num_decisions)
        loggers.agents.debug('[Agent Reward] Cum Reward: %.6f', self.cum_reward)

        return reward

    def calc_reward_backup(self) -> float:
        # !! old way, using next state s_(t+1) to evaluate a_t instead of s_t
        # punishment for non-feasible-action ``past_action_feasible``
        reward: float = 0.0

        if not self.past_action_feasible:
            # non-feasible actions
            # TODO check removal after using action masking
            reward = -100.0
        else:
            # calc reward based on feasible action chosen and utilisation,
            # but based on target station group
            # use last OP because current one already set
            op_rew = self.last_op

            if op_rew is None:
                raise ValueError(
                    ('Tried to calculate reward based on a non-existent operation')
                )
            elif op_rew.target_station_group is None:
                raise ValueError(
                    ('Tried to calculate reward, but no target station group is defined')
                )
            # obtain relevant ProcessingStations contained in
            # the corresponding StationGroup
            stations = op_rew.target_station_group.assoc_proc_stations
            loggers.agents.debug('[AllocationAgent] relevant stations: %s', stations)
            self.set_state_times_current(stations=stations)
            self.calc_utilisation()

            # calculate mean utilisation of all processing stations associated
            # with the corresponding operation and agent's action
            # util_vals: list[float] = [ps.stat_monitor.utilisation for ps in stations]
            util_vals = [self.utilisations[station.system_id] for station in stations]
            loggers.agents.debug('[AllocationAgent] util_vals: %s', util_vals)
            # !! new
            complete_util_vals_all = [
                station.stat_monitor.utilisation for station in self.assoc_proc_stations
            ]
            complete_util_vals_stat_group = [
                station.stat_monitor.utilisation for station in stations
            ]
            loggers.agents.debug(
                '[AllocationAgent] util_vals all: %s', complete_util_vals_all
            )
            loggers.agents.debug(
                '[AllocationAgent] util_vals stat group: %s', complete_util_vals_stat_group
            )
            util_mean = np.mean(util_vals).item()
            complete_util_vals_all_mean = np.mean(complete_util_vals_all).item()
            complete_util_vals_stat_group_mean = np.mean(complete_util_vals_stat_group).item()
            loggers.agents.debug('[AllocationAgent] util_mean: %.4f', util_mean)
            loggers.agents.debug(
                '[AllocationAgent] util_mean_all: %.4f',
                complete_util_vals_all_mean,
            )
            loggers.agents.debug(
                '[AllocationAgent] util_mean_stat_group: %.4f',
                complete_util_vals_stat_group_mean,
            )

            # reward = util_mean - 1.0
            # reward = complete_util_vals_all_mean - 1.0
            reward = complete_util_vals_stat_group_mean - 1.0

        loggers.agents.debug('[AllocationAgent] reward: %.4f', reward)

        return reward


class ValidateAllocationAgent(AllocationAgent):
    def __init__(
        self,
        assoc_system: ProductionArea,
        seed: int | None = None,
    ) -> None:
        super().__init__(assoc_system=assoc_system, seed=seed)

    def simulate_decision_making(self) -> int:
        # sets decision based on allocation policy returns action index
        # use current op
        op = self._current_op
        if op is None:
            raise ValueError('[ALLOC Validation Agent] Operation >>None<<')
        target_stat_group = op.target_station_group
        if target_stat_group is None:
            raise ValueError('[ALLOC Validation Agent] Target station group >>None<<')

        exec_system = self.assoc_system
        target_station = self.env.dispatcher.choose_target_station_from_exec_system(
            exec_system=exec_system,
            target_station_group=target_stat_group,
            is_agent=False,
        )
        # target_station = self.alloc_policy.apply(items=avail_stations)
        action_idx = self.assoc_proc_stations.index(target_station)

        return action_idx


class SequencingAgent(Agent['LogicalQueue[Job]']):
    def __init__(
        self,
        assoc_system: LogicalQueue[Job],
        seed: int | None = None,
    ) -> None:
        super().__init__(assoc_system=assoc_system, agent_task='SEQ', seed=seed)

        # execution control
        # [ACTIONS]
        self.waiting_time = SEQUENCING_WAITING_TIME / self.NORM_TD
        self._relevant_jobs: tuple[Job, ...] | None = None
        self._chosen_job: Job | None = None
        self._req_obj: InfrastructureObject | None = None
        # results
        self.jobs_total: int = 0
        self.jobs_tardy: int = 0
        self.jobs_early: int = 0
        self.jobs_punctual: int = 0

    @property
    def assoc_contents(self) -> tuple[Job, ...]:
        return self.assoc_system.as_tuple()

    @property
    def chosen_job(self) -> Job | None:
        return self._chosen_job

    @override
    def activate(self) -> None:
        pass

    @override
    def request_decision(
        self,
        req_obj: InfrastructureObject,
    ) -> None:
        # indicator that request is being made
        self.set_dispatching_signal(reset=False)
        self._relevant_jobs = self.assoc_contents
        self._req_obj = req_obj
        # build feature vector
        self.feat_vec = self.build_feat_vec(req_object=req_obj)
        loggers.agents.debug(
            '[REQUEST Agent %s]: built FeatVec at %s', self, self.env.t_as_dt()
        )
        loggers.agents.debug('[Agent %s]: Feature Vector: %s', self, self.feat_vec)
        loggers.agents.debug('[Agent %s]: Action Mask: %s', self, self.action_mask)

    @override
    def build_feat_vec(
        self,
        req_object: InfrastructureObject,
    ) -> npt.NDArray[np.float32]:
        action_mask: list[bool] = []
        # station properties
        station_group = cast('StationGroup', req_object.supersystems_as_list()[0])
        res_SGI = station_group.system_id
        avail = int(req_object.stat_monitor.is_available)
        res_info = (res_SGI, avail)
        res_info_arr = np.array(res_info, dtype=np.float32)

        # job properties
        # needed props:
        # - target station group ID
        # - order time
        # - lower bound of slack (set when job was released)
        # - upper bound of slack (set when job was released)
        # - slack (if slack-based rewards used)
        # build feature vector with fixed size, depending on logical queue size
        # --> total vector size: queue size * number of features
        # iterate over all associated jobs and build new vector for each job
        # for each empty place, pad with zeros
        jobs = self._relevant_jobs
        assert jobs is not None, 'relevant jobs not available, >>None<< instead'
        num_q_slots = req_object.logical_queue.size
        num_jobs = len(jobs)
        job_data: list[float] = []
        job_info: tuple[float, ...] = (0, 0, 0, 0, 0)
        action_waiting_feasible: bool = True

        for job in jobs:
            if job.current_order_time is None:
                raise ValueError(f'Current order time of job {job} >>None<<')
            current_op = job.current_op
            if current_op is None:
                raise ValueError(
                    'Tried to build feature vector for job without current operation.'
                )
            order_time = job.current_order_time / self.NORM_TD
            slack_lower_bound = job.stat_monitor.slack_lower_bound_hours
            slack_upper_bound = job.stat_monitor.slack_upper_bound_hours
            slack = job.stat_monitor.slack_hours
            job_SGI = current_op.target_station_group_identifier

            job_info = (
                job_SGI,
                order_time,
                slack_lower_bound,
                slack_upper_bound,
                slack,
            )
            job_data.extend(job_info)

            job_feasible: bool = False
            if res_SGI == job_SGI:
                job_feasible = True
                # waiting action not feasible if any job's slack is
                # below the threshold
                if slack < slack_lower_bound:
                    action_waiting_feasible = False

            action_mask.append(job_feasible)

        feat_vec_len = num_q_slots * len(job_info)
        pad_len = feat_vec_len - len(job_data)
        job_info_arr = np.array(job_data, dtype=np.float32)
        job_info_arr = np.pad(job_info_arr, (0, pad_len), constant_values=0)
        pad_len_actions = num_q_slots - num_jobs
        action_mask_arr = np.array(action_mask, dtype=np.bool_)
        self.action_mask = np.pad(
            action_mask_arr,
            (1, pad_len_actions),
            constant_values=(action_waiting_feasible, False),
        )

        # concat information
        info_arr = np.concatenate((res_info_arr, job_info_arr), dtype=np.float32)

        return info_arr

    @override
    def set_decision(
        self,
        action: int,
    ) -> None:
        self._action = action
        # first action is 'waiting'
        self._chosen_job = None
        if self._relevant_jobs is None:
            raise ValueError('[AGENT][SEQ] Relevant jobs may not be >>None<<')
        loggers.agents.debug(
            '[Agent %s]: Possible jobs are >>%s<<', self, self._relevant_jobs
        )
        loggers.agents.debug(
            '[Agent %s]: Associated queue jobs are >>%s<<', self, self.assoc_contents
        )
        if action != 0:
            if action > len(self._relevant_jobs) and self.env.check_agent_feasibility:
                raise IndexError('Chosen action out of bounds of relevant jobs.')
            elif action > len(self._relevant_jobs):
                # cases in which environment is verified, e.g. by SB3
                action = len(self._relevant_jobs)
            job_idx = action - 1
            self._chosen_job = self._relevant_jobs[job_idx]
        self.num_decisions += 1
        # indicator that request was processed, reset dispatching signal
        self.set_dispatching_signal(reset=True)

        loggers.agents.debug(
            '[DECISION SET Agent %s]: Set to >>%d<<. Chosen job: >>%s<<',
            self,
            self._action,
            self._chosen_job,
        )

    @staticmethod
    def _reward_slack_polynomial(
        x: float,
        a: float = -0.2,
        n: float = 2,
    ) -> float:
        return a * np.power(x, n)

    @staticmethod
    def _reward_curved_norm_scale(
        x: float,
        a: float,
    ) -> float:
        if a <= 0.0:
            raise ValueError('Parameter >>a<< must be greater than 0.')
        a += EPSILON  # eliminate cases where a value of a=1 causes division by zero error
        return (np.power(a, x) - 1) / (a - 1)

    def _reward_slack_category(
        self,
        slack: float,
        slack_upper_bound: float,
        slack_lower_bound: float = 0.0,
    ) -> float:
        # !! What happens if initial slack is smaller than lower bound?
        # !! What happens if current slack is greater than slack_init under the condition
        # !! that slack_init was already tardy?
        # target value needed
        # if a job starts with an initial slack lower than the defined threshold
        # for the upper bound, a target value has to be set
        # every initial slack lower than a given threshold value results in this threshold
        # value as target
        self.jobs_total += 1
        delta: float
        is_target_state: bool = False
        slack_range = abs(slack_upper_bound - slack_lower_bound)
        if slack < slack_lower_bound:
            # tardy
            delta = slack - slack_lower_bound  # negative value
            self.jobs_tardy += 1
        elif slack > slack_upper_bound:
            # premature
            delta = slack - slack_upper_bound
            self.jobs_early += 1
        else:
            # target state: slack_lower_bound <= slack <= slack_upper_bound
            is_target_state = True
            delta = slack - slack_lower_bound  # positive value
            self.jobs_punctual += 1

        proportion = delta / slack_range
        reward: float
        if is_target_state:
            # positive reward: curve between 0 and 1
            reward = self._reward_curved_norm_scale(x=proportion, a=4)
        else:
            # other reward
            reward = self._reward_slack_polynomial(x=proportion)

        return round(reward, ROUNDING_PRECISION)

    @override
    def calc_reward(self) -> float:
        req_obj = self._req_obj
        if req_obj is None:
            raise ValueError('Requesting object is >>None<<.')

        job_reward = self.chosen_job
        loggers.agents.debug('[Agent Reward] Chosen job: %s', job_reward)
        if job_reward is None:
            # reward for waiting
            reward = 0.0
            self.action_feasible = self.action_mask[0]
        else:
            slack_lower_bound = job_reward.stat_monitor.slack_lower_bound_hours
            slack_upper_bound = job_reward.stat_monitor.slack_upper_bound_hours
            slack = job_reward.stat_monitor.slack_hours
            reward = self._reward_slack_category(
                slack=slack,
                slack_lower_bound=slack_lower_bound,
                slack_upper_bound=slack_upper_bound,
            )

            current_op = job_reward.current_op
            if current_op is None:
                raise ValueError(f'Current OP of job {job_reward} >>None<<')
            elif (
                job_reward.time_planned_ending is None
                and current_op.time_planned_ending is None
            ):
                raise ValueError(
                    (
                        'No planned ending dates for neither job nor operation available. '
                        'These are necessary for slack-based rewards.'
                    )
                )
            # obtain target SG from respective operation
            target_station_group = current_op.target_station_group
            # perform feasibility check + set flag
            self.action_feasible = self.env.check_feasible_agent_choice(
                target_station=req_obj, op=current_op
            )
            loggers.agents.debug(
                '[Agent Reward] Target station group: %s', target_station_group
            )

        if not self.action_feasible and self.env.check_agent_feasibility:
            # should never happen with action masking
            raise RuntimeError('Non-feasible action chosen.')

        loggers.agents.debug('[SEQ-Agent] reward: %.4f', reward)
        self.cum_reward += reward
        loggers.agents.debug('[Agent Reward] Num Decisions: %d', self.num_decisions)
        loggers.agents.debug('[Agent Reward] Cum Reward: %.6f', self.cum_reward)

        return reward


class ValidateSequencingAgent(SequencingAgent):
    def __init__(
        self,
        assoc_system: LogicalQueue[Job],
        seed: int | None = None,
    ) -> None:
        super().__init__(assoc_system=assoc_system, seed=seed)

    def simulate_decision_making(self) -> int:
        # sets decision based on allocation policy returns action index
        # use current op
        relevant_jobs = self._relevant_jobs
        req_obj = self._req_obj
        if relevant_jobs is None:
            raise ValueError('[SEQ Validation Agent] No relevant jobs available.')
        if req_obj is None:
            raise ValueError('[SEQ Validation Agent] No requesting object available.')

        job_selected = self.env.dispatcher.choose_job_from_queue(
            req_obj=req_obj,
            queue=self.assoc_system,
            is_agent=False,
            remove_job=False,
        )
        action_idx: int
        if job_selected is None:
            action_idx = 0
        else:
            # offset index by 1 because of waiting action
            action_idx = relevant_jobs.index(job_selected) + 1

        return action_idx
