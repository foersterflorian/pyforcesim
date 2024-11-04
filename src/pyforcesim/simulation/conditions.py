from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.errors import ViolationStartingConditionError
from pyforcesim.simulation.base_components import SimulationComponent
from pyforcesim.types import AgentType

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        SimulationEnvironment,
        Source,
    )

# _dt_mgr = DTManager()


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
        duration_transient: Timedelta,
        name: str = 'TransientCondition',
    ) -> None:
        super().__init__(env=env, name=name)
        # duration after which the condition is set
        self.duration_transient = duration_transient
        self.env.duration_transient = duration_transient

    def pre_process(self) -> None:
        # validate that starting condition is met
        # check transient phase of environment
        if not self.env.is_transient_cond:
            raise ViolationStartingConditionError(
                f'Environment {self.env.name()} not in transient state!'
            )

    def sim_logic(self) -> Generator[None, None, None]:
        sim_time = self.env.td_to_simtime(timedelta=self.duration_transient)
        yield self.sim_control.hold(sim_time, priority=-100)
        # set environment flag and state
        loggers.conditions.debug(
            '[CONDITION %s]: Event list of env: %s', self, self.env._event_list
        )
        self.env.set_end_transient_phase()
        loggers.conditions.info(
            (
                '[CONDITION %s]: Transient Condition over. Set >>is_transient_cond<< '
                'of env to >>%s<<'
            ),
            self,
            self.env.is_transient_cond,
        )
        loggers.conditions.debug(
            '[CONDITION %s]: Event list of env: %s', self, self.env._event_list
        )

    def post_process(self) -> None:
        pass


class JobGenDurationCondition(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        target_obj: Source,
        name: str = 'JobGenDurationCondition',
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
            pyf_dt.validate_dt_UTC(dt=sim_run_until)
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
        yield self.sim_control.hold(sim_time, priority=-100)
        self.target_obj.stop_job_gen_state.set()
        loggers.conditions.info(
            '[CONDITION %s]: Job Generation Condition met at %s', self, self.env.t_as_dt()
        )

    def post_process(self) -> None:
        pass


class TriggerAgentCondition(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        agent: AgentType,
        name: str = 'TriggerAgentCondition',
    ) -> None:
        # initialise base class
        super().__init__(env=env, name=name)

        self.agent = agent

    def pre_process(self) -> None:
        # if self.env.transient_cond_state.get():
        if not self.env.is_transient_cond:
            raise ViolationStartingConditionError(
                (f'Environment {self.env.name()} transient state: State already set!')
            )

    def sim_logic(self) -> Generator[None, None, None]:
        # wait till transient state is over
        if self.env.duration_transient is None:
            raise ValueError('Duration for transient state not set!')
        sim_time = self.env.td_to_simtime(timedelta=self.env.duration_transient)
        yield self.sim_control.hold(sim_time, priority=-90)
        # change allocation rule of dispatcher
        # self.env.dispatcher.alloc_rule = 'AGENT'
        self.agent.assoc_system.trigger_agent_decision()
        loggers.conditions.info(
            '[CONDITION %s]: Set allocation rule to >>AGENT<< at %s', self, self.env.t_as_dt()
        )
        loggers.conditions.debug(
            '[CONDITION %s]: Event list of env: %s', self, self.env._event_list
        )
        # activate agent (trigger needed preparation steps)
        self.agent.activate()

    def post_process(self) -> None:
        pass
