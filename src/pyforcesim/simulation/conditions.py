from __future__ import annotations

import threading
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.errors import ViolationStartingConditionError
from pyforcesim.simulation.base_components import SimulationComponent
from pyforcesim.simulation.environment import ProductionArea, SimulationEnvironment
from pyforcesim.types import AgentType, WIPInputTypes

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        ProductionArea,
        SimulationEnvironment,
        Source,
    )
    from pyforcesim.types import StatDistributionInfo


class BaseCondition(metaclass=ABCMeta):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
    ) -> None:
        self._env = env
        self._name = name
        self.system_id = self.env.get_system_id()

        # [SALABIM COMPONENT]
        self._sim_control = SimulationComponent(
            env=env,
            name=self.name,
            pre_process=self.pre_process,
            sim_logic=self.sim_logic,
            post_process=self.post_process,
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(ID: {self.system_id})'

    def __key(self) -> str:
        return f'{self.__str__()}_{self.system_id}'

    def __hash__(self) -> int:
        return hash(self.__key())

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


class Supervisor(BaseCondition):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
    ) -> None:
        super().__init__(env, name)
        self._stop_execution_event: threading.Event = threading.Event()
        self.supervisors: set[Supervisor] = set()

    @property
    def stop_execution_event(self) -> threading.Event:
        return self._stop_execution_event

    def cancel_external(self) -> None:
        self.sim_control.cancel()

    def register_supervisor(
        self,
        supervisor: Supervisor,
    ) -> None:
        if supervisor not in self.supervisors:
            self.supervisors.add(supervisor)
        else:
            loggers.conditions.warning(
                (
                    'Tried to add supervisor %s to '
                    'supervisor %s, but it was already registered. '
                    ' Nothing done.'
                ),
                supervisor,
                self,
            )

    def stop_execution(self) -> None:
        loggers.conditions.info('[CONDITION] Stopping supervisor >>%s<<', self)
        self.stop_execution_event.set()

        for associated_sv in self.supervisors:
            loggers.conditions.info(
                '[CONDITION] --- Stopping associated supervisor >>%s<<', associated_sv
            )
            associated_sv.stop_execution()
            associated_sv.cancel_external()


class WIPSourceSupervisor(Supervisor):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        *,
        prod_area: ProductionArea,
        supervised_sources: Iterable[Source],
        WIP_relative: float | None = None,
        stat_info_orders: StatDistributionInfo | None = None,
        WIP_time: Timedelta | None = None,
    ) -> None:
        super().__init__(
            env=env,
            name=name,
        )
        # either WIP_relative + stat_info or WIP_Limit
        self.factor_based = bool(WIP_relative and stat_info_orders)
        self.time_based = bool(WIP_time)
        if not any((self.factor_based, self.time_based)):
            raise ValueError(
                (
                    f'[CONDITION] {self}: Either WIP-factor and StatDistributionInfo or '
                    f'WIP-limit must be provided.'
                )
            )
        # factor based WIP limits are preferred
        if self.factor_based:
            self.time_based = False

        self.prod_area = prod_area
        self.supervised_sources = supervised_sources
        self.WIP_relative = WIP_relative
        self.stat_info = stat_info_orders
        self._prod_area_WIP_ideal: Timedelta | None = None
        self.WIP_time = WIP_time
        self._WIP_limit: Timedelta | None = None  # must always be set
        self.sim_priority: float = -98
        self._register_sources()

    @property
    def prod_area_WIP_ideal(self) -> Timedelta:
        if self._prod_area_WIP_ideal is None:
            raise ValueError(
                f'[CONDITION] {self}: Ideal WIP of production area must be set before access.'
            )

        return self._prod_area_WIP_ideal

    @property
    def WIP_limit(self) -> Timedelta:
        if self._WIP_limit is None:
            raise ValueError(f'[CONDITION] {self}: WIP limit must be set before access.')

        return self._WIP_limit

    def _register_sources(self) -> None:
        for source in self.supervised_sources:
            source.register_source_generation_supervisor(self)

    def _get_WIP_ideal(self) -> Timedelta:
        if self.stat_info is None:
            raise ValueError(f'[CONDITION] {self}: Stat-Info is >>None<<.')
        if not self.prod_area.initialised:
            self.prod_area.initialise()

        return self.prod_area.WIP_ideal(self.stat_info)

    def _change_depending_parameters(self, WIP_relative: float) -> None:
        for source in self.supervised_sources:
            for job_generator in source.job_generators:
                job_generator.recalculate_expected_arrival_time(WIP_relative)
                job_generator.recalculate_lead_time_delta(WIP_relative)

    def _set_WIP_relative(
        self,
        WIP_relative: float,
    ) -> Timedelta:
        if not WIP_relative > 0:
            raise ValueError(f'[CONDITION] {self}: WIP factor must be greater than 0.')
        WIP_limit = self.prod_area_WIP_ideal * WIP_relative

        return WIP_limit

    def _set_WIP_time(
        self,
        WIP_time: Timedelta,
    ) -> Timedelta:
        if not WIP_time > Timedelta():
            raise ValueError(f'[CONDITION] {self}: WIP limit must be greater than 0.')
        WIP_limit = WIP_time

        return WIP_limit

    def change_WIP_limit(
        self,
        WIP_relative_set: float | None = None,
        WIP_time_set: Timedelta | None = None,
    ) -> None:
        WIP_limit: Timedelta
        if self.factor_based and WIP_relative_set is None:
            raise ValueError(
                f'[CONDITION] {self}: Factor based WIP calculation needs new WIP-relative.'
            )
        elif self.factor_based and WIP_relative_set is not None:
            WIP_limit = self._set_WIP_relative(WIP_relative_set)
            self._change_depending_parameters(WIP_relative_set)
        elif self.time_based and WIP_time_set is None:
            raise ValueError(f'[CONDITION] {self}: calculation needs new WIP-limit.')
        elif self.time_based and WIP_time_set is not None:
            WIP_limit = self._set_WIP_time(WIP_time_set)
        else:
            RuntimeError(
                (
                    f'[CONDITION] {self}: Wrong combination of used parameter '
                    f'to set and underlying WIP limit type. WIP limit type: '
                    f'factor={self.factor_based}, limit(Timedelta)={self.time_based}, '
                    f'type of variable factor={type(WIP_relative_set).__name__}, '
                    f'limit={type(WIP_time_set).__name__}'
                )
            )

        self._WIP_limit = WIP_limit

        loggers.conditions.debug(
            '[CONDITION] %s: Set WIP Limit to >>%s<< at %s',
            self,
            self._WIP_limit,
            self.env.t_as_dt(),
        )

    def allow_production(self) -> bool:
        _, WIP_current, _ = self.prod_area.get_content_WIP()
        # loggers.conditions.debug(
        #     '[CONDITION] %s: WIP = %s at %s',
        #     self,
        #     WIP_current,
        #     self.env.t_as_dt(),
        # )
        production_allowed: bool
        if WIP_current >= self.WIP_limit:
            # loggers.conditions.debug(
            #     '[CONDITION] %s: Stop production at %s',
            #     self,
            #     self.env.t_as_dt(),
            # )
            production_allowed = False
        else:
            # loggers.conditions.debug(
            #     '[CONDITION] %s: Start production at %s',
            #     self,
            #     self.env.t_as_dt(),
            # )
            production_allowed = True

        return production_allowed

    @override
    def pre_process(self) -> None:
        if self.factor_based:
            self._prod_area_WIP_ideal = self._get_WIP_ideal()

        self.change_WIP_limit(self.WIP_relative, self.WIP_time)

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        yield self.sim_control.hold(0)

    @override
    def post_process(self) -> None:
        pass


class WIPSourceSupervisorLimitSetter(Supervisor):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        sim_interval: Timedelta,
        *,
        WIP_source_supervisor: WIPSourceSupervisor,
        WIP_relatives: Sequence[float] | None = None,
        WIP_times: Sequence[Timedelta] | None = None,
    ) -> None:
        super().__init__(env=env, name=name)
        self.sim_interval = sim_interval

        self.target_WIP_supervisor = WIP_source_supervisor
        self.target_WIP_supervisor.register_supervisor(self)
        WIP_inputs = self._verify_WIP_inputs(WIP_relatives=WIP_relatives, WIP_times=WIP_times)

        self._WIP_relatives: tuple[float, ...] | None = None
        self._WIP_times: tuple[Timedelta, ...] | None = None
        self.WIP_limits_max_idx: int
        if WIP_inputs.factors:
            self._WIP_relatives = WIP_inputs.factors
            self.WIP_limits_max_idx = len(self._WIP_relatives) - 1
            self._check_WIP_relatives()
        elif WIP_inputs.times:
            self._WIP_times = WIP_inputs.times
            self.WIP_limits_max_idx = len(self._WIP_times) - 1
            self._check_WIP_times()

        self.WIP_limits_curr_idx: int | None = None
        self.sim_interval_time_units: float | None = None
        # higher priority than associated WIP controller
        self.sim_priority = self.target_WIP_supervisor.sim_priority - 1

    @property
    def WIP_relatives(self) -> tuple[float, ...]:
        if self._WIP_relatives is None:
            raise ValueError(f'[CONDITION] {self}: WIP factors must be set before access.')

        return self._WIP_relatives

    @property
    def WIP_times(self) -> tuple[Timedelta, ...]:
        if self._WIP_times is None:
            raise ValueError(f'[CONDITION] {self}: WIP limits must be set before access.')

        return self._WIP_times

    def _verify_WIP_inputs(
        self,
        WIP_relatives: Sequence[float] | None,
        WIP_times: Sequence[Timedelta] | None,
    ) -> WIPInputTypes:
        factor_based = self.target_WIP_supervisor.factor_based
        time_based = self.target_WIP_supervisor.time_based
        factors: tuple[float, ...] = tuple()
        times: tuple[Timedelta, ...] = tuple()

        if factor_based and WIP_relatives is None:
            raise ValueError(
                (
                    f'[CONDITION] {self}: Parent supervisor is factor-based, '
                    f'but no factors were provided.'
                )
            )
        elif factor_based and WIP_relatives is not None:
            factors = tuple(WIP_relatives)
        elif time_based and WIP_times is None:
            raise ValueError(
                (
                    f'[CONDITION] {self}: Parent supervisor is limit-based, '
                    f'but no limits were provided.'
                )
            )
        elif time_based and WIP_times is not None:
            times = tuple(WIP_times)
        else:
            # should never be triggered
            RuntimeError(
                (
                    f'[CONDITION] {self}: Wrong combination of used parameter '
                    f'to set and underlying WIP limit type. WIP limit type: '
                    f'factor={factor_based}, limit(Timedelta)={time_based}, '
                    f'type of variable factor={type(WIP_relatives).__name__}, '
                    f'limit={type(WIP_times).__name__}'
                )
            )

        return WIPInputTypes(factors=factors, times=times)

    def _check_WIP_relatives(self) -> None:
        WIP_lower_bound: float = 0

        for idx, factor in enumerate(self.WIP_relatives):
            if not factor > WIP_lower_bound:
                raise ValueError(
                    (
                        f'[CONDITION] {self}: WIP factor {factor} with '
                        f'index {idx} must be greater than 0.'
                    )
                )

    def _check_WIP_times(self) -> None:
        WIP_lower_bound: Timedelta = Timedelta()

        for idx, limit in enumerate(self.WIP_times):
            if not limit > WIP_lower_bound:
                raise ValueError(
                    (
                        f'[CONDITION] {self}: WIP limit {limit} with '
                        f'index {idx} must be greater than 0.'
                    )
                )

    def _get_WIP_limit_idx(self) -> int:
        """cycle through WIP limits. If end is reached, start in front again

        Returns
        -------
        int
            current relevant WIP limit index
        """
        if self.WIP_limits_curr_idx is None:
            self.WIP_limits_curr_idx = 0
            return self.WIP_limits_curr_idx

        new_idx = self.WIP_limits_curr_idx + 1
        if new_idx > self.WIP_limits_max_idx:
            new_idx = 0
        self.WIP_limits_curr_idx = new_idx

        return self.WIP_limits_curr_idx

    def _set_WIP_limit(self) -> None:
        """set the current relevant WIP limit for the associated target
        WIP controller
        """
        limit_idx = self._get_WIP_limit_idx()
        loggers.conditions.debug(
            '[CONDITION] %s: Current WIP limit index is %d at %s',
            self.__class__.__name__,
            limit_idx,
            self.env.t_as_dt(),
        )

        factor_to_set: float | None = None
        time_to_set: Timedelta | None = None
        if self.target_WIP_supervisor.factor_based:
            factor_to_set = self.WIP_relatives[limit_idx]
        elif self.target_WIP_supervisor.time_based:
            time_to_set = self.WIP_times[limit_idx]
        else:
            raise RuntimeError(
                f'[CONDITION] {self}: Neither factor nor limit based WIP limits.'
            )

        self.target_WIP_supervisor.change_WIP_limit(
            WIP_relative_set=factor_to_set,
            WIP_time_set=time_to_set,
        )
        loggers.conditions.info(
            '[CONDITION] %s: Set WIP limit to: factor=>%s<, time=>%s< at %s',
            self.__class__.__name__,
            factor_to_set,
            time_to_set,
            self.env.t_as_dt(),
        )

    @override
    def pre_process(self) -> None:
        if not self.sim_interval > Timedelta():
            raise ValueError(
                (
                    f'[CONDITION] {self.__class__.__name__}: Simulation interval '
                    f'must be greater than 0.'
                )
            )
        self.sim_interval_time_units = self.env.td_to_simtime(self.sim_interval)
        # set initial WIP limit
        self._set_WIP_limit()

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        assert (
            self.sim_interval_time_units is not None
        ), 'Interval in simulation time units must not be >>None<<'

        loggers.conditions.info(
            '[CONDITION] %s: WIP_relatives = %s, WIP_limits = %s',
            self.__class__.__name__,
            self._WIP_relatives,
            self._WIP_times,
        )
        # initial WIP limit set in pre-process method
        while not self.target_WIP_supervisor.stop_execution_event.is_set():
            yield self.sim_control.hold(
                self.sim_interval_time_units, priority=self.sim_priority
            )
            self._set_WIP_limit()  # set new WIP limit

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

        self.observers: set[Observer] = set()

    def register_observer(self, observer: Observer) -> None:
        if observer not in self.observers:
            self.observers.add(observer)
        else:
            loggers.conditions.warning(
                (
                    'Tried to add observer %s to '
                    'observer %s, but it was already registered. '
                    ' Nothing done.'
                ),
                observer,
                self,
            )

    def cancel_external(self) -> None:
        self.sim_control.cancel()

    @property
    def stop_execution_event(self) -> threading.Event:
        return self._stop_execution_event

    def stop_execution(self) -> None:
        loggers.conditions.info('[CONDITION] Stopping observer >>%s<<', self)
        self.stop_execution_event.set()

        for associated_obs in self.observers:
            loggers.conditions.info(
                '[CONDITION] --- Stopping associated observer >>%s<<', associated_obs
            )
            associated_obs.stop_execution()
            associated_obs.cancel_external()


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
        self.sim_priority: float = -98

        self._register_sources()

    def _register_sources(self) -> None:
        for source in self.target_sources:
            source.register_source_generation_controller(self)

    def change_WIP_limit(
        self,
        new_limit: Timedelta,
    ) -> None:
        if not new_limit > Timedelta():
            raise ValueError(
                '[CONDITION] WIPSourceController: WIP limit must be greater than 0.'
            )
        self.WIP_limit = new_limit

    @override
    def pre_process(self) -> None:
        if not self.sim_interval > Timedelta():
            raise ValueError(
                '[CONDITION] WIPSourceController: Simulation interval must be greater than 0.'
            )
        if not self.WIP_limit > Timedelta():
            raise ValueError(
                '[CONDITION] WIPSourceController: WIP limit must be greater than 0.'
            )
        self.sim_interval_time_units = self.env.td_to_simtime(self.sim_interval)

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        assert (
            self.sim_interval_time_units is not None
        ), 'Interval in simulation time units must not be >>None<<'

        loggers.conditions.info(
            '[CONDITION] WIPSourceController: WIP_limit = %s', self.WIP_limit
        )

        while not self.stop_execution_event.is_set():
            _, WIP_current, _ = self.prod_area.get_content_WIP()
            loggers.conditions.debug(
                '[CONDITION] WIPSourceController: WIP = %s at %s',
                WIP_current,
                self.env.t_as_dt(),
            )
            if WIP_current >= self.WIP_limit and not self.stop_production.is_set():
                loggers.conditions.debug(
                    '[CONDITION] WIPSourceController: Stop production at %s',
                    self.env.t_as_dt(),
                )
                self.stop_production.set()
            elif WIP_current < self.WIP_limit and self.stop_production.is_set():
                loggers.conditions.debug(
                    '[CONDITION] WIPSourceController: Start production at %s',
                    self.env.t_as_dt(),
                )
                self.stop_production.clear()

            yield self.sim_control.hold(
                self.sim_interval_time_units, priority=self.sim_priority
            )

    @override
    def post_process(self) -> None:
        pass


class WIPLimitSetter(Observer):
    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        sim_interval: Timedelta,
        *,
        WIP_source_controller: WIPSourceController,
        WIP_limits: Sequence[Timedelta],
    ) -> None:
        super().__init__(
            env=env,
            name=name,
            setting_event=threading.Event(),  # placeholder, not needed
            sim_interval=sim_interval,
        )

        self.target_WIP_controller = WIP_source_controller
        self.target_WIP_controller.register_observer(self)

        self.WIP_limits = tuple(WIP_limits)
        self.WIP_limits_max_idx = len(self.WIP_limits) - 1
        self.WIP_limits_curr_idx: int | None = None
        self.sim_interval_time_units: float | None = None
        # higher priority than associated WIP controller
        self.sim_priority = self.target_WIP_controller.sim_priority - 1

    def _check_WIP_limits(self) -> None:
        WIP_lower_bound = Timedelta()

        for idx, limit in enumerate(self.WIP_limits):
            if not limit > WIP_lower_bound:
                raise ValueError(
                    (
                        f'[CONDITION] WIPLimitSetter: WIP limit {limit} with '
                        f'index {idx} must be greater than 0.'
                    )
                )

    def _get_WIP_limit_idx(self) -> int:
        """cycle through WIP limits. If end is reached, start in front again

        Returns
        -------
        int
            current relevant WIP limit index
        """
        if self.WIP_limits_curr_idx is None:
            self.WIP_limits_curr_idx = 0
            return self.WIP_limits_curr_idx

        new_idx = self.WIP_limits_curr_idx + 1
        if new_idx > self.WIP_limits_max_idx:
            new_idx = 0
        self.WIP_limits_curr_idx = new_idx

        return self.WIP_limits_curr_idx

    def _set_WIP_limit(self) -> None:
        """set the current relevant WIP limit for the associated target
        WIP controller
        """
        limit_idx = self._get_WIP_limit_idx()
        loggers.conditions.debug(
            '[CONDITION] WIPLimitSetter: Current WIP limit index is %d at %s',
            limit_idx,
            self.env.t_as_dt(),
        )

        limit_to_set = self.WIP_limits[limit_idx]
        self.target_WIP_controller.change_WIP_limit(limit_to_set)
        loggers.conditions.info(
            '[CONDITION] WIPLimitSetter: Set WIP limit to %s at %s',
            limit_to_set,
            self.env.t_as_dt(),
        )

    @override
    def pre_process(self) -> None:
        self._check_WIP_limits()
        if not self.sim_interval > Timedelta():
            raise ValueError(
                '[CONDITION] WIPLimitSetter: Simulation interval must be greater than 0.'
            )
        self.sim_interval_time_units = self.env.td_to_simtime(self.sim_interval)
        # set initial WIP limit
        self._set_WIP_limit()

    @override
    def sim_logic(self) -> Generator[None, None, None]:
        assert (
            self.sim_interval_time_units is not None
        ), 'Interval in simulation time units must not be >>None<<'

        loggers.conditions.info(
            '[CONDITION] WIPLimitSetter: WIP_limits = %s', self.WIP_limits
        )
        # initial WIP limit set in pre-process method
        while not self.target_WIP_controller.stop_execution_event.is_set():
            yield self.sim_control.hold(
                self.sim_interval_time_units, priority=self.sim_priority
            )
            self._set_WIP_limit()  # set new WIP limit

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
