from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any

from pyforcesim import loggers
from pyforcesim.datetime import DTManager
from pyforcesim.errors import ViolationStartingConditionError
from pyforcesim.simulation.base_components import SimulationComponent

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        SimulationEnvironment,
        Source,
    )

_dt_mgr = DTManager()


class BaseCondition(ABC):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
    ) -> None:
        self._env = env
        self._name = name

        # [SALABIM COMPONENT]
        self._sim_control = SimulationComponent(
            env=env,
            name=self.name,
            pre_process=self.pre_process,
            sim_logic=self.sim_logic,
            post_process=self.post_process,
        )

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def name(self) -> str:
        return self._name

    @property
    def sim_control(self) -> SimulationComponent:
        return self._sim_control

    @abstractmethod
    def pre_process(self) -> Any:
        """returns: tuple with values or None"""
        ...

    @abstractmethod
    def sim_logic(self) -> Generator[Any, Any, Any]:
        """returns: tuple with values or None"""
        ...

    @abstractmethod
    def post_process(self) -> Any:
        """returns: tuple with values or None"""
        ...


class TransientCondition(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        duration_transient: Timedelta,
    ) -> None:
        super().__init__(env=env, name=name)
        # duration after which the condition is set
        self.duration_transient = duration_transient

    def pre_process(self) -> None:
        # validate that starting condition is met
        # check transient phase of environment
        if not self.env.is_transient_cond:
            raise ViolationStartingConditionError(
                f'Environment {self.env.name()} not in transient state!'
            )

    def sim_logic(self) -> Generator[None, None, None]:
        sim_time = self.env.td_to_simtime(timedelta=self.duration_transient)
        yield self.sim_control.hold(sim_time, priority=-10)
        # set environment flag and state
        self.env.is_transient_cond = False
        self.env.transient_cond_state.set()
        loggers.conditions.info(
            (
                f'[CONDITION {self}]: Transient Condition over. Set >>is_transient_cond<< '
                f'of env to >>{self.env.is_transient_cond}<<'
            )
        )

    def post_process(self) -> None:
        pass


class JobGenDurationCondition(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        target_obj: Source,
        sim_run_until: Datetime | None = None,
        sim_run_duration: Timedelta | None = None,
    ) -> None:
        # initialise base class
        super().__init__(env=env, name=name)

        # either point in time or duration must be provided
        if not any((sim_run_until, sim_run_duration)):
            raise ValueError('Either point in time or duration must be provided')
        elif all((sim_run_until, sim_run_duration)):
            raise ValueError('Point in time and duration provided. Only one allowed.')

        starting_dt = self.env.starting_datetime
        self.sim_run_duration: Timedelta
        if sim_run_until is not None:
            _dt_mgr.validate_dt_UTC(dt=sim_run_until)
            if sim_run_until <= starting_dt:
                raise ValueError(
                    (
                        'Point in time must not be in the past. '
                        f'Starting Time Env: {starting_dt}, '
                        f'Point in time provided: {sim_run_until}'
                    )
                )
            self.sim_run_duration = sim_run_until - starting_dt
        elif sim_run_duration is not None:
            self.sim_run_duration = sim_run_duration
        else:
            raise ValueError('No valid point in time or duration provided')

        self.sim_run_until = sim_run_until
        self.target_obj = target_obj
        self.target_obj.stop_job_gen_cond_reg = True

    def pre_process(self) -> None:
        if self.target_obj.stop_job_gen_state.get():
            raise ViolationStartingConditionError(
                f'Target object {self.target_obj}: Flag not unset!'
            )

    def sim_logic(self) -> Generator[None, None, None]:
        sim_time = self.env.td_to_simtime(timedelta=self.sim_run_duration)
        yield self.sim_control.hold(sim_time, priority=-10)
        self.target_obj.stop_job_gen_state.set()
        loggers.conditions.info(
            (f'[CONDITION {self}]: Job Generation Condition met at ' f'{self.env.t_as_dt()}')
        )

    def post_process(self) -> None:
        pass


class TriggerAgentCondition(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
    ) -> None:
        # initialise base class
        super().__init__(env=env, name=name)

    def pre_process(self) -> None:
        if self.env.transient_cond_state.get():
            raise ViolationStartingConditionError(
                (f'Environment {self.env.name()} ' 'transient state: State already set!')
            )

    def sim_logic(self) -> Generator[None, None, None]:
        # wait till transient state is over
        yield self.sim_control.wait(self.env.transient_cond_state, priority=-9)
        # change allocation rule of dispatcher
        self.env.dispatcher.curr_alloc_rule = 'AGENT'
        loggers.conditions.info((f'[CONDITION {self}]: Set allocation rule to >>AGENT<<'))

    def post_process(self) -> None:
        pass
