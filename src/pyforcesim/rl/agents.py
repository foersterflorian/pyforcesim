from __future__ import annotations

import random
import statistics
from abc import ABC, abstractmethod
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from pyforcesim import loggers
from pyforcesim.types import AgentTasks

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        Job,
        Operation,
        ProcessingStation,
        SimulationEnvironment,
        System,
    )
    from pyforcesim.simulation.monitors import InfStructMonitor


class Agent(ABC):
    def __init__(
        self,
        assoc_system: 'System',
        agent_task: AgentTasks,
    ) -> None:
        # basic information
        self._agent_task = agent_task
        # associated system
        self._assoc_system, self._env = assoc_system.register_agent(
            agent=self, agent_task=self._agent_task
        )
        # dispatching signal: no matter if allocation or sequencing
        self._dispatching_signal: bool = False

    def __str__(self) -> str:
        return f'Agent(type={self._agent_task}, Assoc_Syst_ID={self._assoc_system.system_id})'

    @property
    def assoc_system(self) -> 'System':
        return self._assoc_system

    @property
    def agent_task(self) -> str:
        return self._agent_task

    @property
    def env(self) -> 'SimulationEnvironment':
        return self._env

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
            f'Dispatching signal for >>{self}<< was set to >>{self._dispatching_signal}<<.'
        )

    @abstractmethod
    def request_decision(self) -> Any:
        pass

    @abstractmethod
    def set_decision(self) -> Any:
        pass

    @abstractmethod
    def build_feat_vec(self) -> Any:
        pass

    @abstractmethod
    def calc_reward(self) -> Any:
        pass


class AllocationAgent(Agent):
    def __init__(
        self,
        assoc_system: 'System',
    ) -> None:
        # init base class
        super().__init__(assoc_system=assoc_system, agent_task='ALLOC')

        # get associated systems
        self._assoc_proc_stations = self._assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )

        # job-related properties
        self._current_job: 'Job | None' = None
        self._last_job: 'Job | None' = None
        self._current_op: 'Operation | None' = None
        self._last_op: 'Operation | None' = None

        # RL related properties
        self.feat_vec: npt.NDArray[np.float32] | None = None

        # execution control
        # [ACTIONS]
        # action chosen by RL agent
        self._action: int | None = None
        self.action_feasible: bool = False
        self.past_action_feasible: bool = False

    @property
    def current_job(self) -> 'Job | None':
        return self._current_job

    @property
    def last_job(self) -> 'Job | None':
        return self._last_job

    @property
    def current_op(self) -> 'Operation | None':
        return self._current_op

    @property
    def last_op(self) -> 'Operation | None':
        return self._last_op

    @property
    def action(self) -> int | None:
        return self._action

    @property
    def assoc_proc_stations(self) -> tuple['ProcessingStation', ...]:
        return self._assoc_proc_stations

    def update_assoc_proc_stations(self) -> None:
        # get associated systems
        self._assoc_proc_stations = self._assoc_system.lowest_level_subsystems(
            only_processing_stations=True
        )

    def request_decision(
        self,
        job: 'Job',
        op: 'Operation',
    ) -> None:
        # for each request, decision not done yet
        # indicator for internal loop
        # self._RL_decision_done = False
        # set flag indicating an request was made
        # indicator for external loop in Gym Env
        # self._RL_decision_request = True

        # indicator that request is being made
        self.set_dispatching_signal(reset=False)

        # remember relevant jobs
        if job != self._current_job:
            self._last_job = self._current_job
            self._current_job = job
        if op != self._current_op:
            self._last_op = self._current_op
            self._current_op = op

        # build feature vector
        self.feat_vec = self.build_feat_vec(job=job)

        loggers.agents.debug(f'[REQUEST Agent {self}]: built FeatVec.')

    def set_decision(
        self,
        action: int | None,
    ) -> None:
        # get action from RL agent
        self._action = action
        # indicator that request was processed
        # reset dispatching signal
        self.set_dispatching_signal(reset=True)

        loggers.agents.debug(f'[DECISION SET Agent {self}]: Set {self._action=}')

    # ?? REWORK necessary?
    def build_feat_vec(
        self,
        job: 'Job',
    ) -> npt.NDArray[np.float32]:
        # resources
        # needed properties
        # station group, availability, WIP_time
        for i, res in enumerate(self._assoc_proc_stations):
            # T1 build feature vector for one machine
            # !! type of stats monitor not clear
            monitor = cast('InfStructMonitor', res.stat_monitor)
            # station group identifier should be the system's one
            # because custom IDs can be non-numeric which is bad for an agent
            # use only first identifier although multiple values are possible
            res_sys_SGI = list(res.supersystems_ids)[0]
            # availability: boolean to integer
            avail = int(monitor.is_available)
            # WIP_time in hours
            WIP_time = monitor.WIP_load_time / Timedelta(hours=1)
            # tuple: (System SGI of resource obj, availability status,
            # WIP calculated in time units)
            res_info = (res_sys_SGI, avail, WIP_time)
            res_info_arr = np.array(res_info, dtype=np.float32)

            if i == 0:
                arr = res_info_arr
            else:
                arr = np.concatenate((arr, res_info_arr))

        # job
        # needed properties
        # target station group ID, order time
        assert job.current_order_time is not None
        order_time = job.current_order_time / Timedelta(hours=1)
        # current op: obtain StationGroupID
        current_op = job.current_op
        if current_op is not None:
            job_SGI = current_op.target_station_group_identifier
            assert job_SGI is not None
        else:
            raise ValueError(
                ('Tried to build feature vector for job without ' 'current operation.')
            )
        # TODO: remove, internal identifiers now all SystemIDs
        """
        # SGI is type CustomID, but system ID (SystemID) is needed
        # lookup system ID by custom ID in Infrastructure Manager
        infstruct_mgr = self.env.infstruct_mgr
        system_id = infstruct_mgr.lookup_system_ID(
            subsystem_type='StationGroup',
            custom_ID=job_SGI,
        )
        """
        job_info = (job_SGI, order_time)
        job_info_arr = np.array(job_info, dtype=np.float32)

        # concat job information
        arr = np.concatenate((arr, job_info_arr))

        return arr

    def random_action(self) -> int:
        """
        Generate random action based on associated objects
        """
        return random.randint(0, len(self._assoc_proc_stations) - 1)

    def calc_reward(self) -> float:
        # !! REWORK
        # TODO change reward type hint
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
                # catch empty OPs
                raise ValueError(
                    ('Tried to calculate reward based ' 'on a non-existent operation')
                )
            elif op_rew.target_station_group is None:
                # catch empty OPs
                raise ValueError(
                    ('Tried to calculate reward, but no ' 'target station group is defined')
                )
            # obtain relevant ProcessingStations contained in
            # the corresponding StationGroup
            stations = op_rew.target_station_group.assoc_proc_stations
            loggers.agents.debug(f'++++++ {stations=}')
            # calculate mean utilisation of all processing stations associated
            # with the corresponding operation and agent's action
            # !! inheritance scheme not sufficient, couple of type mismatches arising
            util_vals: list[float] = [ps.stat_monitor.utilisation for ps in stations]
            loggers.agents.debug(f'++++++ {util_vals=}')
            util_mean = statistics.mean(util_vals)
            loggers.agents.debug(f'++++++ {util_mean=}')

            reward = util_mean - 1.0

        loggers.agents.debug(f'+#+#+#+#+# {reward=}')

        return reward


class SequencingAgent(Agent):
    def __init__(
        self,
        assoc_system: 'System',
    ) -> None:
        raise NotImplementedError('SequencingAgent not implemented yet.')
        # super().__init__(assoc_system=assoc_system, agent_task='SEQ')
