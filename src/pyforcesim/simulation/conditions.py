from __future__ import annotations

import threading
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterable
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.errors import ViolationStartingConditionError
from pyforcesim.simulation.base_components import SimulationComponent
from pyforcesim.simulation.environment import ProductionArea, SimulationEnvironment
from pyforcesim.types import AgentType

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        ProductionArea,
        SimulationEnvironment,
        Source,
    )


class BaseCondition(metaclass=ABCMeta):
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

    @override
    def pre_process(self) -> None:
        # validate that starting condition is met
        # check transient phase of environment
        if not self.env.is_transient_cond:
            raise ViolationStartingConditionError(
                f'Environment {self.env.name()} not in transient state!'
            )

    @override
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

    @override
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

    @override
    def pre_process(self) -> None:
        if self.target_obj.stop_job_gen_state.get():
            raise ViolationStartingConditionError(
                f'Target object {self.target_obj}: Flag not unset!'
            )

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        sim_time = self.env.td_to_simtime(timedelta=self.sim_run_duration)
        yield self.sim_control.hold(sim_time, priority=-100)
        self.target_obj.stop_job_gen_state.set()
        loggers.conditions.info(
            '[CONDITION %s]: Job Generation Condition met at %s', self, self.env.t_as_dt()
        )

    @override
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

    @override
    def pre_process(self) -> None:
        # if self.env.transient_cond_state.get():
        if not self.env.is_transient_cond:
            raise ViolationStartingConditionError(
                (f'Environment {self.env.name()} transient state: State already set!')
            )

    @override
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

    @override
    def post_process(self) -> None:
        pass


class Observer(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        setting_event: threading.Event,
        sim_interval: Timedelta,
    ) -> None:
        super().__init__(env, name)
        self.setting_event = setting_event
        self._stop_execution_event = threading.Event()
        self.sim_interval = sim_interval
        self.system_id = self.env.get_system_id()

    def __key(self) -> str:
        return f'{self.__str__()}_{self.system_id}'

    def __hash__(self) -> int:
        return hash(self.__key())

    @property
    def stop_execution_event(self) -> threading.Event:
        return self._stop_execution_event

    def stop_execution(self) -> None:
        loggers.conditions.info('Stopping observer >>%s<<', self)
        self.stop_execution_event.set()


class WIPSourceController(Observer):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        setting_event: threading.Event,
        sim_interval: Timedelta,
        *,
        prod_area: ProductionArea,
        target_sources: Iterable[Source],
        WIP_limit: Timedelta,
    ) -> None:
        super().__init__(
            env=env,
            name=name,
            setting_event=setting_event,
            sim_interval=sim_interval,
        )

        self.stop_production = self.setting_event
        self.prod_area = prod_area
        self.target_sources = target_sources
        self.WIP_limit = WIP_limit
        self.sim_interval_time_units: float | None = None

        self._register_sources()

    def _register_sources(self) -> None:
        for source in self.target_sources:
            source.register_source_generation_controller(self)

    @override
    def pre_process(self) -> None:
        if not self.sim_interval > Timedelta():
            raise ValueError(
                '[CONDITION] WIP-Controller: Simulation interval must be greater than 0.'
            )
        if not self.WIP_limit > Timedelta():
            raise ValueError('[CONDITION] WIP-Controller: WIP limit must be greater than 0.')
        self.sim_interval_time_units = self.env.td_to_simtime(self.sim_interval)

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        assert (
            self.sim_interval_time_units is not None
        ), 'Interval in simulation time units must not be >>None<<'

        loggers.conditions.info('[CONDITION] WIP-Controller: WIP_limit = %s', self.WIP_limit)

        while not self.stop_execution_event.is_set():
            _, WIP_current, _ = self.prod_area.get_content_WIP()
            loggers.conditions.debug(
                '[CONDITION] WIP-Controller: WIP = %s at %s', WIP_current, self.env.t_as_dt()
            )
            if WIP_current >= self.WIP_limit and not self.stop_production.is_set():
                loggers.conditions.debug(
                    '[CONDITION] Stop production at %s', self.env.t_as_dt()
                )
                self.stop_production.set()
            elif WIP_current < self.WIP_limit and self.stop_production.is_set():
                loggers.conditions.debug(
                    '[CONDITION] Start production at %s', self.env.t_as_dt()
                )
                self.stop_production.clear()

            yield self.sim_control.hold(self.sim_interval_time_units, priority=-98)

    @override
    def post_process(self) -> None:
        pass


# class ObserverThread(threading.Thread, metaclass=ABCMeta):
#     def __init__(
#         self,
#         env: SimulationEnvironment,
#         name: str,
#         setting_event: threading.Event,
#     ) -> None:
#         super().__init__(group=None, target=None, name=name, daemon=True)
#         self._env = env
#         self._sim_name = name
#         self.setting_event = setting_event
#         self._stop_execution_event = threading.Event()

#     def __str__(self) -> str:
#         return f'{self.__class__.__name__}(Name: {self.sim_name})'

#     @property
#     def env(self) -> SimulationEnvironment:
#         return self._env

#     @property
#     def sim_name(self) -> str:
#         return self._sim_name

#     @property
#     def stop_execution_event(self) -> threading.Event:
#         return self._stop_execution_event

#     @abstractmethod
#     def sim_logic(self) -> None: ...

#     @override
#     def run(self) -> None:
#         loggers.conditions.info('Starting observer >>%s<<', self)
#         self.sim_logic()

#     def stop_execution(self) -> None:
#         loggers.conditions.info('Stopping observer >>%s<<', self)
#         self.stop_execution_event.set()


# class WIPSourceControllerThread(ObserverThread):
#     def __init__(
#         self,
#         env: SimulationEnvironment,
#         name: str,
#         setting_event: threading.Event,
#         *,
#         prod_area: ProductionArea,
#         target_sources: Iterable[Source],
#         WIP_limit: Timedelta,
#     ) -> None:
#         super().__init__(env=env, name=name, setting_event=setting_event)
#         self.stop_production = self.setting_event
#         self.prod_area = prod_area
#         self.target_sources = target_sources
#         self.WIP_limit = WIP_limit

#         self._register_sources()

#     def _register_sources(self) -> None:
#         for source in self.target_sources:
#             source.register_source_generation_event(self.stop_production)

#     @override
#     def sim_logic(self) -> None:
#         while not self.stop_execution_event.is_set():
#             _, WIP_current, _ = self.prod_area.get_content_WIP()
#             if WIP_current >= self.WIP_limit:
#                 loggers.conditions.info(
#                     '--------[CONDITION] Stop production at %s', self.env.t_as_dt()
#                 )
#                 self.stop_production.set()
#             elif WIP_current < self.WIP_limit and self.stop_production.is_set():
#                 loggers.conditions.info(
#                     '--------[CONDITION] Start production at %s', self.env.t_as_dt()
#                 )
#                 self.stop_production.clear()

#             time.sleep(0.5)
