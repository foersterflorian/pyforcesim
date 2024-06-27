from __future__ import annotations

import random
import statistics
from abc import ABC, abstractmethod
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np
import numpy.typing as npt

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.constants import HELPER_STATES, UTIL_PROPERTIES, TimeUnitsTimedelta
from pyforcesim.types import AgentTasks, StateTimes, SystemID

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        Job,
        Operation,
        ProcessingStation,
        SimulationEnvironment,
        StationGroup,
        System,
    )


class Agent(ABC):
    def __init__(
        self,
        assoc_system: System,
        agent_task: AgentTasks,
        seed: int = 42,
    ) -> None:
        # basic information
        self._agent_task = agent_task
        # associated system
        self._assoc_system, self._env = assoc_system.register_agent(
            agent=self, agent_task=self._agent_task
        )
        # dispatching signal: no matter if allocation or sequencing
        self._dispatching_signal: bool = False

        self._rng = random.Random(seed)

    def __str__(self) -> str:
        return f'Agent(type={self._agent_task}, Assoc_Syst_ID={self._assoc_system.system_id})'

    @property
    def assoc_system(self) -> System:
        return self._assoc_system

    @property
    def agent_task(self) -> str:
        return self._agent_task

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def dispatching_signal(self) -> bool:
        return self._dispatching_signal

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
    def request_decision(self) -> None: ...

    @abstractmethod
    def set_decision(self) -> None: ...

    @abstractmethod
    def build_feat_vec(self) -> npt.NDArray[np.float32]: ...

    @abstractmethod
    def calc_reward(self) -> float: ...


class AllocationAgent(Agent):
    def __init__(
        self,
        assoc_system: System,
        seed: int = 42,
    ) -> None:
        # init base class
        super().__init__(assoc_system=assoc_system, agent_task='ALLOC', seed=seed)

        # get associated systems
        self._assoc_proc_stations = self._assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )
        # default action mask: allow all actions
        self._action_mask: npt.NDArray[np.bool_] = np.tile(
            [True], len(self._assoc_proc_stations)
        )

        # job-related properties
        self._current_job: Job | None = None
        self._last_job: Job | None = None
        self._current_op: Operation | None = None
        self._last_op: Operation | None = None

        # states
        self.state_times_last: dict[SystemID, StateTimes] = {}
        self.state_times_current: dict[SystemID, StateTimes] = {}
        self.utilisations: dict[SystemID, float] = {}

        for station in self.assoc_proc_stations:
            self.state_times_last[station.system_id] = station.stat_monitor.state_times.copy()
            self.state_times_current[station.system_id] = (
                station.stat_monitor.state_times.copy()
            )
            self.utilisations[station.system_id] = 0.0

        # RL related properties
        self.feat_vec: npt.NDArray[np.float32] | None = None

        # execution control
        # [ACTIONS]
        # action chosen by RL agent
        self._action: int | None = None
        self._action_feasible: bool = False
        self.past_action_feasible: bool = False
        # count number of consecutive non-feasible actions
        self.non_feasible_counter: int = 0

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
    def action(self) -> int | None:
        return self._action

    @property
    def assoc_proc_stations(self) -> tuple[ProcessingStation, ...]:
        return self._assoc_proc_stations

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
        self._action_feasible = feasible

    @property
    def action_mask(self) -> npt.NDArray[np.bool_]:
        return self._action_mask

    @action_mask.setter
    def action_mask(
        self,
        val: npt.NDArray[np.bool_],
    ) -> None:
        if len(val) != len(self._action_mask):
            raise ValueError('Action mask length does not match number of resources.')
        self._action_mask = val

    def update_assoc_proc_stations(self) -> None:
        # get associated systems
        self._assoc_proc_stations = self._assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )

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

        # if time_total.total_seconds() > 0:
        if time_non_helpers.total_seconds() > 0:
            # utilisation = round(time_utilisation / time_total, 4)
            utilisation = round(time_utilisation / time_non_helpers, 4)

        return utilisation

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

        for station in self.assoc_proc_stations:
            sys_id = station.system_id
            self.state_times_last[sys_id] = self.state_times_current[sys_id].copy()
            self.state_times_current[sys_id] = station.stat_monitor.state_times.copy()
            # calculate difference between last and current for each station
            # calculate utilisation (difference based) for each station
            utilisation = self.calc_util_state_time_diff(
                state_times_last=self.state_times_last[sys_id],
                state_times_current=self.state_times_current[sys_id],
            )
            self.utilisations[sys_id] = utilisation
            loggers.agents.debug('Utilisation for SystemID %d is %.4f', sys_id, utilisation)

        loggers.agents.debug('Utilisation dict for agent >>%s<<: %s', self, self.utilisations)

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
        # get action from RL agent
        self._action = action
        # indicator that request was processed
        # reset dispatching signal
        self.set_dispatching_signal(reset=True)

        loggers.agents.debug('[DECISION SET Agent %s]: Set %d', self, self._action)

    @override
    def build_feat_vec(
        self,
        job: Job,
    ) -> npt.NDArray[np.float32]:
        action_mask: list[bool] = []
        station_feasible: bool
        # job
        # needed properties
        # target station group ID, order time
        if job.current_order_time is None:
            raise ValueError(f'Current order time of job {job} >>None<<')
        norm_td = pyf_dt.timedelta_from_val(1.0, TimeUnitsTimedelta.HOURS)
        order_time = job.current_order_time / norm_td
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
        job_info = (job_SGI, order_time)
        job_info_arr = np.array(job_info, dtype=np.float32)
        # resources
        # needed properties
        # station group, availability, WIP_time
        for i, res in enumerate(self._assoc_proc_stations):
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
            WIP_time = monitor.WIP_load_time / Timedelta(hours=1)
            # tuple: (System SGI of resource obj, availability status,
            # WIP calculated in time units)
            res_info = (res_SGI, avail, WIP_time)
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

    def random_action(self) -> int:
        """
        Generate random action based on associated objects
        """
        return self.rng.randint(0, len(self._assoc_proc_stations) - 1)

    @override
    def calc_reward(self) -> float:
        # punishment for non-feasible-action ``past_action_feasible``
        reward: float = 0.0

        if not self.past_action_feasible:
            # non-feasible actions
            reward = -100.0
        else:
            # calc reward based on feasible action chosen and
            # utilisation, but based on target station group
            # use last OP because current one already set
            op_rew = self._last_op

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
            loggers.agents.debug('relevant stations: %s', stations)

            # relevant_state_times: list[dict[str, Timedelta]] = []
            # util_vals: list[float] = []
            # for station in stations:
            #     # self.state_times_current[station.system_id] = (
            #     #     station.stat_monitor.state_times.copy()
            #     # )
            #     util_vals.append(self.utilisations[station.system_id])
            #     # last state times
            #     # current state times

            # calculate mean utilisation of all processing stations associated
            # with the corresponding operation and agent's action
            # util_vals: list[float] = [ps.stat_monitor.utilisation for ps in stations]
            util_vals: list[float] = [
                self.utilisations[station.system_id] for station in stations
            ]
            loggers.agents.debug('util_vals: %s', util_vals)
            util_mean = statistics.mean(util_vals)
            loggers.agents.debug('util_mean: %.4f', util_mean)

            reward = util_mean - 1.0

        loggers.agents.debug('reward: %.4f', reward)

        return reward


class SequencingAgent(Agent):
    def __init__(
        self,
        assoc_system: System,
    ) -> None:
        raise NotImplementedError('SequencingAgent not implemented yet.')
        # super().__init__(assoc_system=assoc_system, agent_task='SEQ')
