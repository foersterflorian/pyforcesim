from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, Generic, TypeVar, cast
from typing_extensions import override

import pandas as pd
import plotly.express as px
from pandas import DataFrame, Series

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.common import enum_str_values_as_frzset
from pyforcesim.constants import (
    EPSILON,
    HELPER_STATES,
    INF,
    PROCESSING_PROPERTIES,
    SLACK_ADAPTION,
    SLACK_ADAPTION_MIN_LOWER_BOUND,
    SLACK_ADAPTION_MIN_UPPER_BOUND,
    SLACK_DEFAULT_LOWER_BOUND,
    SLACK_INIT_AS_UPPER_BOUND,
    SLACK_MAX_RANGE,
    SLACK_MIN_RANGE,
    SLACK_OVERWRITE_UPPER_BOUND,
    SLACK_THRESHOLD_UPPER,
    SLACK_USE_THRESHOLD_UPPER,
    UTIL_PROPERTIES,
    SimStatesAvailability,
    SimStatesCommon,
    SimStatesStorage,
    TimeUnitsTimedelta,
)
from pyforcesim.types import LoadObjects, MonitorObjects

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        InfrastructureObject,
        Job,
        Operation,
        SimulationEnvironment,
        StationGroup,
        StorageLike,
    )
    from pyforcesim.types import (
        LoadID,
        PlotlyFigure,
    )

T = TypeVar('T', bound=MonitorObjects)
L = TypeVar('L', bound=LoadObjects)


class Monitor(Generic[T]):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: T,
        current_state: SimStatesCommon | SimStatesStorage = SimStatesCommon.INIT,
        states: type[SimStatesCommon | SimStatesStorage] = SimStatesCommon,
    ) -> None:
        """
        Class to monitor associated objects (load and resource)
        """
        # [REGISTRATION]
        self._env = env
        self._target_object = obj
        self.NORM_TD: Final[Timedelta] = pyf_dt.timedelta_from_val(
            1.0, TimeUnitsTimedelta.HOURS
        )

        if current_state == SimStatesCommon.TEMP or current_state == SimStatesStorage.TEMP:
            raise ValueError('TEMP state is not allowed as initial state.')

        # [STATE] state parameters
        self.states_possible = enum_str_values_as_frzset(states)
        # check integrity of the given state
        self.state_current = current_state

        # boolean indicator if a state is set
        self.state_status: dict[str, bool] = {}
        # time counter for each state
        self.state_times: dict[str, Timedelta] = {}
        # starting time variable indicating when the last state assignment took place
        self.state_starting_time = self.env.t_as_dt()

        for state in self.states_possible:
            self.state_times[state] = Timedelta()
            if state == self.state_current:
                self.state_status[state] = True
            else:
                self.state_status[state] = False

        # DataFrame to further analyse state durations
        self.state_durations: DataFrame | None = None
        # availability indicator
        self._availability_states = enum_str_values_as_frzset(SimStatesAvailability)
        if (
            self.state_current in self._availability_states
            or self.state_current == SimStatesCommon.INIT
        ):
            self.is_available: bool = True
        else:
            self.is_available: bool = False

        # additional 'TEMP' state information
        # indicator if state was 'TEMP'
        self._is_temp: bool = False
        # state before 'TEMP' was set
        self._state_before_temp = self.state_current
        # time components
        self.time_total: Timedelta = Timedelta()
        self.time_non_helpers: Timedelta = Timedelta()
        self.time_utilisation: Timedelta = Timedelta()
        self.time_processing: Timedelta = Timedelta()

        # time handling
        # loggers.monitors.debug('Monitor states: %s', self.states_possible)
        # loggers.monitors.debug('Monitor state times: %s', self.state_times)

    def __repr__(self) -> str:
        return f'Monitor instance of {self.target_object}'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def target_object(self) -> T:
        return self._target_object

    def get_current_state(self) -> SimStatesCommon | SimStatesStorage:
        """get the current state of the associated resource"""
        return self.state_current

    def set_state(
        self,
        target_state: SimStatesCommon | SimStatesStorage,
    ) -> None:
        """
        function to set the object in the given state
        state: name of the state in which the object should be placed, must be part \
            of the object's possible states
        """

        # validity check
        if target_state not in self.states_possible:
            raise ValueError(
                (
                    f'The state {target_state} is not allowed. '
                    f'Must be one of {self.states_possible}'
                )
            )

        # check if state is already set
        if self.state_status[target_state] and target_state != SimStatesCommon.TEMP:
            loggers.monitors.debug(
                'Tried to set state of %s to >>%s<<, but this state was already set.'
                ' State of object was not changed.',
                self.target_object,
                target_state,
            )
        # check if the 'TEMP' state was already set, this should never happen
        # if it happens raise an error to catch wrong behaviour
        elif self.state_status[target_state] and target_state == SimStatesCommon.TEMP:
            raise RuntimeError(
                (
                    f'Tried to set state of {self._target_object} to >>TEMP<<, '
                    f'but this state was already set.'
                )
            )

        # calculate time for which the object was in the current state before changing it
        current_state = self.state_current
        current_state_start = self.state_starting_time
        current_time = self._env.t_as_dt()
        current_state_duration = current_time - current_state_start
        self.state_times[current_state] += current_state_duration

        # check if 'TEMP' state shall be set
        if target_state == SimStatesCommon.TEMP:
            # set 'TEMP' state indicator to true
            self._is_temp = True
            # save current state for the state reset
            self._state_before_temp = current_state

        # set old state to False and new state to True
        self.state_status[current_state] = False
        self.state_status[target_state] = True
        # assign new state as current one
        self.state_current = target_state
        self.state_starting_time = current_time
        # availability
        if self.state_current in self._availability_states:
            self.is_available: bool = True
        elif self.state_current == SimStatesStorage.TEMP:
            # 'TEMP' state shall not change the availability indicator
            pass
        else:
            self.is_available: bool = False

        loggers.monitors.debug(
            'Duration for state %s on %s was %s',
            current_state,
            self.target_object,
            current_state_duration,
        )

    def reset_temp_state(self) -> None:
        """Reset from 'TEMP' state"""
        # check if object was in TEMP state, raise error if not
        if self._is_temp:
            self._is_temp = False
            self.set_state(target_state=self._state_before_temp)
        else:
            raise RuntimeError(
                (
                    f'Tried to reset {self._target_object} from >>TEMP<< state but '
                    f'the current state is >>{self.state_current}<<'
                )
            )

    def _KPI_time_proportions(
        self,
    ) -> None:
        calc_utilisation: bool = False
        if hasattr(self, 'utilisation'):
            calc_utilisation = True

        time_total = Timedelta()
        time_non_helpers = Timedelta()
        time_utilisation = Timedelta()
        time_processing = Timedelta()

        for state, duration in self.state_times.items():
            time_total += duration
            if state not in HELPER_STATES:
                time_non_helpers += duration
            if state in PROCESSING_PROPERTIES:
                time_processing += duration
            if calc_utilisation and state in UTIL_PROPERTIES:
                time_utilisation += duration

        self.time_total = time_total
        self.time_non_helpers = time_non_helpers
        self.time_processing = time_processing
        if calc_utilisation:
            self.time_utilisation = time_utilisation

    def calc_KPI(self) -> None:
        """calculates different KPIs at any point in time"""
        # state durations for analysis
        self._KPI_time_proportions()

    def state_durations_as_df(self) -> DataFrame:
        """Calculates absolute and relative state durations at the current time

        Returns
        -------
        DataFrame
            State duration table with absolute and relative values
        """
        # build state duration table
        # loggers.monitors.debug(
        #     'State durations for %s: %s',
        #     self.target_object,
        #     self.state_times,
        # )
        temp1: Series = pd.Series(data=self.state_times)
        temp2: DataFrame = temp1.to_frame()
        temp2.columns = ['abs [Timedelta]']
        temp2['abs [seconds]'] = temp2['abs [Timedelta]'].apply(
            func=lambda x: x.total_seconds()
        )
        temp2['rel [%]'] = temp2['abs [seconds]'] / temp2.sum(axis=0)['abs [seconds]'] * 100.0
        drop_labels = list(HELPER_STATES)
        temp2 = temp2.drop(labels=drop_labels, axis=0)
        temp2 = temp2.sort_index(axis=0, ascending=True, kind='stable')
        state_durations_df = temp2.copy()

        return state_durations_df

    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # assign state duration table
        self.state_durations = self.state_durations_as_df()
        # calculate KPIs
        self.calc_KPI()

    ### ANALYSE AND CHARTS ###
    def draw_state_chart(
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'state_distribution',
        time_unit: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
        pie_chart: bool = False,
    ) -> PlotlyFigure:
        """draws the collected state times of the object as bar chart"""
        data = pd.DataFrame.from_dict(
            data=self.state_times, orient='index', columns=['total time']
        )
        data.index = data.index.rename('state')
        # change time from Timedelta to any time unit possible --> float
        # Plotly can not handle Timedelta objects properly, only Datetimes
        calc_td = pyf_dt.timedelta_from_val(val=1.0, time_unit=time_unit)
        calc_col: str = f'total time [{time_unit}]'
        data[calc_col] = data['total time'] / calc_td  # type: ignore
        data = data.sort_index(axis=0, kind='stable')

        show_legend: bool
        chart_type: str
        if pie_chart:
            data = data.loc[data[calc_col] > 0.0, :]
            fig = px.pie(data, values=calc_col, names=data.index)
            show_legend = True
            chart_type = 'Pie'
        else:
            fig = px.bar(data, y=calc_col, text_auto='.2f')  # type: ignore wrong type hint in Plotly
            show_legend = False
            chart_type = 'Bar'

        fig.update_layout(
            title=f'State Time Distribution of {self._target_object}', showlegend=show_legend
        )
        fig.update_yaxes(title=dict({'text': calc_col}))

        fig.show()

        file_name = (
            file_name
            + f'_{chart_type}_{self.target_object.__class__.__name__}'
            + f'_CustomID_{self.target_object.custom_identifier}'
        )
        save_path = Path.cwd() / 'results' / file_name

        if save_html:
            save_path_html = save_path.with_suffix('.html')
            fig.write_html(save_path_html)

        if save_img:
            save_path_img = save_path.with_suffix('.svg')
            fig.write_image(save_path_img)

        return fig


class LoadMonitor(Monitor[L]):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: L,
        current_state: SimStatesCommon = SimStatesCommon.INIT,
    ) -> None:
        super().__init__(
            env=env,
            obj=obj,
            current_state=current_state,
            states=SimStatesCommon,
        )
        self._released: bool = False
        self.remaining_order_time = self.target_object.order_time
        self.slack_planned: Timedelta | None = None
        self.slack: Timedelta = Timedelta()
        self.slack_init: Timedelta = Timedelta()
        self.slack_init_hours: float = 0.0
        self.slack_upper_bound: Timedelta = Timedelta()
        self.slack_upper_bound_init: Timedelta | None = None
        self.slack_upper_bound_adaption_delta: Timedelta = Timedelta()
        self.slack_upper_bound_hours: float = 0.0
        self.slack_lower_bound: Timedelta = SLACK_DEFAULT_LOWER_BOUND
        self.slack_lower_bound_hours: float = SLACK_DEFAULT_LOWER_BOUND / self.NORM_TD

    @property
    def released(self) -> bool:
        return self._released

    def _init_slack(self) -> None:
        self.calc_KPI()
        self.slack_init = self.slack
        self.slack_init_hours = self.slack_hours

        self.slack_upper_bound = self.slack_init
        loggers.monitors.debug(
            '[MONITOR]: Slack upper bound >%s< after init: >%s< ',
            self.slack_upper_bound,
            self.slack_init,
        )

        if (
            SLACK_INIT_AS_UPPER_BOUND
            and SLACK_USE_THRESHOLD_UPPER
            and self.slack_upper_bound < SLACK_THRESHOLD_UPPER
        ):
            self.slack_upper_bound = SLACK_THRESHOLD_UPPER
        elif not SLACK_INIT_AS_UPPER_BOUND:
            if self.slack_upper_bound > SLACK_OVERWRITE_UPPER_BOUND:
                self.slack_upper_bound = SLACK_OVERWRITE_UPPER_BOUND

        self.assert_slack_range()
        self.slack_upper_bound_hours = self.slack_upper_bound / self.NORM_TD
        self.slack_upper_bound_init = self.slack_upper_bound

    def assert_slack_range(self) -> None:
        if (
            self.slack_upper_bound <= self.slack_lower_bound
            or abs(self.slack_upper_bound - self.slack_lower_bound) < SLACK_MIN_RANGE
        ):
            loggers.monitors.debug(
                (
                    '[MONITOR]: Slack lower bound before (min) range adaption: >%s< '
                    '(%.4f), upper bound: >%s<, slack_init: %s'
                ),
                self.slack_lower_bound,
                self.slack_lower_bound_hours,
                self.slack_upper_bound,
                self.slack_init,
            )
            # upper bound is at least "adaption_min_upper_bound" cfg option
            lower_bound_adapted = self.slack_upper_bound - SLACK_MIN_RANGE
            # ?? keep hard slack min bound or use ranges in future?
            # self.slack_lower_bound = max(lower_bound_adapted, SLACK_ADAPTION_MIN_LOWER_BOUND)
            self.slack_lower_bound = lower_bound_adapted
            self.slack_lower_bound_hours = self.slack_lower_bound / self.NORM_TD
            loggers.monitors.debug(
                (
                    '[MONITOR]: Slack lower bound adapted for min range: >%s< '
                    '(%.4f), upper bound: (%.4f), min_range: %s'
                ),
                self.slack_lower_bound,
                self.slack_lower_bound_hours,
                self.slack_upper_bound_hours,
                SLACK_MIN_RANGE,
            )

        if abs(self.slack_upper_bound - self.slack_lower_bound) > SLACK_MAX_RANGE:
            loggers.monitors.debug(
                (
                    '[MONITOR]: Slack lower bound before (max) range adaption: >%s< '
                    '(%.4f), upper bound: >%s<, slack_init: %s'
                ),
                self.slack_lower_bound,
                self.slack_lower_bound_hours,
                self.slack_upper_bound,
                self.slack_init,
            )
            # upper bound is at least "adaption_min_upper_bound" cfg option
            self.slack_lower_bound = self.slack_upper_bound - SLACK_MAX_RANGE
            self.slack_lower_bound_hours = self.slack_lower_bound / self.NORM_TD
            loggers.monitors.debug(
                (
                    '[MONITOR]: Slack lower bound adapted for max range: >%s< '
                    '(%.4f), upper bound: (%.4f), max_range: %s'
                ),
                self.slack_lower_bound,
                self.slack_lower_bound_hours,
                self.slack_upper_bound_hours,
                SLACK_MAX_RANGE,
            )

    def release(self) -> None:
        """certain actions performed on release"""
        self._init_slack()
        self._released = True

    def slack_time_units(
        self,
        time_unit: TimeUnitsTimedelta,
    ) -> float:
        return self.slack / pyf_dt.timedelta_from_val(1.0, time_unit)

    @property
    def slack_hours(self) -> float:
        return self.slack_time_units(TimeUnitsTimedelta.HOURS)


class JobMonitor(LoadMonitor['Job']):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: Job,
        current_state: SimStatesCommon = SimStatesCommon.INIT,
    ) -> None:
        super().__init__(
            env=env,
            obj=obj,
            current_state=current_state,
        )
        self.slack_range_init: Timedelta | None = None
        self._slack_considered_ops: set[LoadID] = set()

    def _init_slack_range(
        self,
        lower_bound: Timedelta,
    ) -> None:
        self.slack_planned = Timedelta()
        for op in self.target_object.operations:
            assert (
                op.stat_monitor.slack_planned is not None
            ), 'tried summation of planned slack times from which at least one was not set'
            self.slack_planned += op.stat_monitor.slack_planned

        self.slack_range_init = self.slack_planned - lower_bound

    def _KPI_remaining_order_time(self) -> None:
        # !! make sure that all open operations' KPIs are calculated
        remaining_order_time: Timedelta = Timedelta()
        for op in self.target_object.open_operations:
            remaining_order_time += op.stat_monitor.remaining_order_time
        # !! make sure that time proportions are calculated
        self.remaining_order_time = remaining_order_time

        loggers.operations.debug(
            '[%s] Calculated remaining order time as %s, total: %s at %s',
            self,
            self.remaining_order_time,
            self.target_object.order_time,
            self.env.t_as_dt(),
        )

    def _KPI_slack(self) -> None:
        time_planned_ending = self.target_object.time_planned_ending
        if time_planned_ending is not None:
            curr_time = self.env.t_as_dt()
            time_till_due = time_planned_ending - curr_time
            # !! make sure that remaining order time is calculated
            self.slack = time_till_due - self.remaining_order_time

            loggers.jobs.debug(
                '[%s] Calculated slack as %.4f h at %s. Planned ending date: %s',
                self,
                self.slack_hours,
                curr_time,
                time_planned_ending,
            )

    def _total_slack_upper_bound_adaption(self) -> None:
        if not SLACK_ADAPTION:
            raise RuntimeError(
                f'Tried to adapt slack of {self.target_object} even though slack '
                f'adaption is not enbaled.'
            )
        if not self.released:
            # only with release status all slack parameters initialised
            return

        op = self.target_object.current_op
        if op is None:
            return
        elif not op.stat_monitor.slack_adapted:
            return
        elif op.op_id in self._slack_considered_ops:
            return
        # consecutively add adaption delta of all operations
        self.slack_upper_bound_adaption_delta += (
            op.stat_monitor.slack_upper_bound_adaption_delta
        )
        self._slack_considered_ops.add(op.op_id)

        loggers.monitors.debug(
            '[MONITOR]: OpID: %d Slack - Adaption delta of OP is >%s<, new total delta: >%s<',
            op.op_id,
            op.stat_monitor.slack_upper_bound_adaption_delta,
            self.slack_upper_bound_adaption_delta,
        )
        loggers.monitors.debug(
            '[MONITOR]: JobID: %d Slack UB before adaption >%s<',
            self.target_object.job_id,
            self.slack_upper_bound,
        )

        assert (
            self.slack_upper_bound_init is not None
        ), f'init upper bound not set for JobID: {self.target_object.job_id}'
        self.slack_upper_bound = (
            self.slack_upper_bound_init + self.slack_upper_bound_adaption_delta
        )
        self.assert_slack_range()
        self.slack_upper_bound_hours = self.slack_upper_bound / self.NORM_TD

        loggers.monitors.debug(
            '[MONITOR]: JobID: %d Slack - Adaption delta is >%s<, new upper bound is >%s<',
            self.target_object.job_id,
            self.slack_upper_bound_adaption_delta,
            self.slack_upper_bound,
        )

    @override
    def release(self) -> None:
        super().release()
        self._init_slack_range(SLACK_DEFAULT_LOWER_BOUND)

    @override
    def calc_KPI(self) -> None:
        super().calc_KPI()  # especially time proportions
        for op in self.target_object.open_operations:
            op.stat_monitor.calc_KPI()
        self._KPI_remaining_order_time()
        if SLACK_ADAPTION:
            self._total_slack_upper_bound_adaption()
        self._KPI_slack()


class OperationMonitor(LoadMonitor['Operation']):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: Operation,
        current_state: SimStatesCommon = SimStatesCommon.INIT,
    ) -> None:
        super().__init__(
            env=env,
            obj=obj,
            current_state=current_state,
        )
        self._slack_adapted: bool = False

    @property
    def slack_adapted(self) -> bool:
        return self._slack_adapted

    def init_slack_planned(
        self,
        by_date: bool = True,
    ) -> None:
        if by_date:
            self.slack_planned = self._init_slack_planned_by_date()
        else:
            self.slack_planned = self._init_slack_planned_by_lead_time()

    def _init_slack_planned_by_date(self) -> Timedelta:
        planned_start_date = self.target_object.time_planned_starting
        planned_end_date = self.target_object.time_planned_ending

        if planned_start_date is None or planned_end_date is None:
            raise ValueError(
                'Can not calculate slack range by dates if dates are not provided.'
            )

        planned_lead_time = planned_end_date - planned_start_date
        slack_range_init = planned_lead_time - self.target_object.order_time

        return slack_range_init

    def _init_slack_planned_by_lead_time(self) -> Timedelta:
        exec_system = self.target_object.target_exec_system
        planned_lead_time = exec_system.lead_time_planned
        slack_range_init = planned_lead_time - self.target_object.order_time

        return slack_range_init

    def _KPI_remaining_order_time(self) -> None:
        time_actual_starting = self.target_object.time_actual_starting
        total_order_time = self.target_object.order_time
        if time_actual_starting is not None:
            # !! make sure that time proportions are calculated
            self.remaining_order_time = total_order_time - self.time_processing

        loggers.operations.debug(
            '[%s] [Load-ID: %d] Calculated remaining order time as %s, total: %s at %s',
            self,
            self.target_object.op_id,
            self.remaining_order_time,
            self.target_object.order_time,
            self.env.t_as_dt(),
        )

    def _KPI_slack(self) -> None:
        time_planned_ending = self.target_object.time_planned_ending
        if time_planned_ending is not None:
            curr_time = self.env.t_as_dt()
            time_till_due = time_planned_ending - curr_time
            # !! make sure that remaining order time is calculated
            self.slack = time_till_due - self.remaining_order_time

            loggers.operations.debug(
                '[%s] Calculated slack as %.4f h at %s. Planned ending date: %s',
                self,
                self.slack_hours,
                curr_time,
                time_planned_ending,
            )

    def _adapt_slack(self) -> None:
        if self.slack_adapted:
            # perform slack adaption only once for OPs
            return

        prod_area = self.target_object.target_exec_system
        lead_time_delta = prod_area.lead_time_delta

        if lead_time_delta != Timedelta():
            upper_bound_calc = self.slack_upper_bound + lead_time_delta
            # ?? use fixed min upper bound?
            # upper_bound_adapted = max(upper_bound_calc, SLACK_ADAPTION_MIN_UPPER_BOUND)
            upper_bound_adapted = upper_bound_calc
            self.slack_upper_bound_adaption_delta += (
                upper_bound_adapted - self.slack_upper_bound
            )
            self.slack_upper_bound = upper_bound_adapted
            self.assert_slack_range()
            self.slack_upper_bound_hours = self.slack_upper_bound / self.NORM_TD

            loggers.monitors.debug(
                '[MONITOR][Ops] ID: %d Slack: upper bound calc >%s<, upper bound adapted >%s<',
                self.target_object.op_id,
                upper_bound_calc,
                upper_bound_adapted,
            )
            loggers.monitors.debug(
                '[MONITOR][Ops] OP(%s), ID: %d Slack: upper bound adaption delta >%s<',
                self.target_object,
                self.target_object.op_id,
                self.slack_upper_bound_adaption_delta,
            )

        self._slack_adapted = True

    @override
    def release(self) -> None:
        super().release()
        if SLACK_ADAPTION:
            self._adapt_slack()

    @override
    def calc_KPI(self) -> None:
        super().calc_KPI()  # especially time proportions
        self._KPI_remaining_order_time()  # needed for slack
        self._KPI_slack()


class StorageMonitor(Monitor['StorageLike']):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: StorageLike,
        current_state: SimStatesStorage = SimStatesStorage.INIT,
    ) -> None:
        # initialise parent class
        super().__init__(
            env=env,
            obj=obj,
            current_state=current_state,
            states=SimStatesStorage,
        )

        # fill level tracking
        self._level_db_types = {
            'sim_time': object,
            'duration': object,
            'level': int,
        }
        self._level_db: DataFrame = pd.DataFrame(
            columns=['sim_time', 'duration', 'level'],
            data=[[self.env.t_as_dt(), Timedelta(), obj.fill_level_init]],
        )
        self._level_db = self._level_db.astype(self._level_db_types)

        self._current_fill_level = obj.fill_level_init
        self._fill_level_starting_time: Datetime = self.env.t_as_dt()
        self._wei_avg_fill_level: float | None = None
        # WIPs
        self.contents_WIP_load_time: Timedelta = Timedelta()
        self.contents_WIP_load_time_remaining: Timedelta = Timedelta()
        self.contents_WIP_load_num_jobs: int = 0

    @property
    def wei_avg_fill_level(self) -> float | None:
        return self._wei_avg_fill_level

    @property
    def level_db(self) -> DataFrame:
        return self._level_db

    @override
    def set_state(
        self,
        target_state: SimStatesStorage,
    ) -> None:
        """additional level tracking functionality"""
        super().set_state(target_state=target_state)

        is_finalise: bool = False
        if self.state_current == SimStatesCommon.FINISH:
            is_finalise = True

        self._calc_contents_WIP()
        self._track_fill_level(is_finalise=is_finalise)

    def _calc_contents_WIP(self) -> None:
        relevant_store = cast(Iterable['Job'], self.target_object.sim_control.store)
        WIP_num: int = 0
        WIP_load_time: Timedelta = Timedelta()
        for job in relevant_store:
            current_op = job.current_op
            if current_op is None:
                raise ValueError(
                    f'Current OP >>None<< during WIP retrieval in StorageMonitor >>{self}<<.'
                )
            WIP_load_time += current_op.order_time
            WIP_num += 1

        self.contents_WIP_load_time = WIP_load_time
        self.contents_WIP_load_time_remaining = WIP_load_time
        self.contents_WIP_load_num_jobs = WIP_num

    # storage fill level tracking
    def _track_fill_level(
        self,
        is_finalise: bool = False,
    ) -> None:
        """adds an entry to the fill level database"""
        # only calculate duration if buffer level changes
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        duration: Timedelta = current_time - self._fill_level_starting_time
        loggers.buffers.debug(
            '[BUFFER: %s] Current time is %s with level %s and old level %s',
            self.target_object,
            current_time,
            self.target_object.fill_level,
            self._current_fill_level,
        )
        # if ((self._current_fill_level != len(self)) and (duration > 0.0)) or is_finalise:
        if self._current_fill_level != self.target_object.fill_level or is_finalise:
            temp1: Series = pd.Series(
                index=['sim_time', 'duration', 'level'],
                data=[current_time, duration, self._current_fill_level],
            )
            temp2: DataFrame = temp1.to_frame().T.astype(self._level_db_types)
            self._level_db = pd.concat([self._level_db, temp2], ignore_index=True)
            self._current_fill_level = self.target_object.fill_level
            self._fill_level_starting_time = current_time

    @override
    def calc_KPI(self) -> None:
        super().calc_KPI()
        self._calc_contents_WIP()

    @override
    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # execute parent class function
        super().finalise_stats()

        # finalise fill level tracking
        self._track_fill_level(is_finalise=True)

        # weighted average fill level
        self._level_db = self._level_db.loc[
            self._level_db['duration'] > Timedelta(), :
        ].copy()
        self._level_db = self._level_db.reset_index(drop=True)
        temp1: DataFrame = self._level_db.copy()
        temp1['duration_seconds'] = temp1['duration'].apply(func=lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        # sums: Series = temp1.sum(axis=0)
        self._wei_avg_fill_level = cast(
            float, sums['mul'] / (sums['duration_seconds'] + EPSILON)
        )

    ### ANALYSE AND CHARTS ###
    def draw_fill_level(
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'fill_level',
    ) -> PlotlyFigure:
        """
        method to draw and display the fill level expansion of the corresponding buffer
        """
        # add starting point to start chart at t = init time
        data = self._level_db.copy()
        val1: float = data.at[0, 'sim_time'] - data.at[0, 'duration']
        val2: float = 0.0
        val3: int = data.at[0, 'level']
        temp1: DataFrame = pd.DataFrame(columns=data.columns, data=[[val1, val2, val3]])
        temp1 = pd.concat([temp1, data], ignore_index=True)

        fig = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape='vh')
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title=f'Fill Level of {self._target_object}')
        fig.update_yaxes(title=dict({'text': 'fill level [-]'}))
        fig.update_xaxes(title=dict({'text': 'time'}))
        # weighted average fill level
        fig.add_hline(
            y=self.wei_avg_fill_level, line_width=3, line_dash='dot', line_color='orange'
        )
        # capacity
        cap = self._target_object.capacity
        if cap < INF:
            fig.add_hline(y=cap, line_width=3, line_dash='dash', line_color='red')

        fig.show()

        file_name = file_name + f'_{self}'

        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)

        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)

        return fig


class InfStructMonitor(Monitor['InfrastructureObject']):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: InfrastructureObject,
        current_state: SimStatesCommon = SimStatesCommon.INIT,
    ) -> None:
        # initialise parent class
        super().__init__(
            env=env,
            obj=obj,
            current_state=current_state,
            states=SimStatesCommon,
        )
        # WIP tracking time load
        self._WIP_time_db_types = {
            'sim_time': object,
            'duration': object,
            'level': object,
        }
        # ?? PERHAPS ADD STARTING LEVEL LATER
        self._WIP_time_db: DataFrame = pd.DataFrame(
            columns=['sim_time', 'duration', 'level'],
            data=[[self.env.t_as_dt(), Timedelta(), Timedelta()]],
        )
        self._WIP_time_db = self._WIP_time_db.astype(self._WIP_time_db_types)

        # WIP tracking number of jobs
        self._WIP_num_db_types = {
            'sim_time': object,
            'duration': object,
            'level': int,
        }
        # ?? PERHAPS ADD STARTING LEVEL LATER
        self._WIP_num_db: DataFrame = pd.DataFrame(
            columns=['sim_time', 'duration', 'level'],
            data=[[self.env.t_as_dt(), Timedelta(), 0]],
        )
        self._WIP_num_db = self._WIP_num_db.astype(self._WIP_num_db_types)

        self._WIP_time_starting_time: Datetime = self.env.t_as_dt()
        self._WIP_num_starting_time: Datetime = self.env.t_as_dt()
        self._wei_avg_WIP_level_time: Timedelta | None = None
        self._wei_avg_WIP_level_num: float | None = None

        # time components
        self.time_occupied: float = 0.0

        # resource KPIs
        self.utilisation: float = 0.0
        self.time_op_processed: Timedelta = Timedelta()
        self.time_op_remaining: Timedelta = Timedelta()

        # logistic objective values
        self.WIP_load_time: Timedelta = Timedelta()
        self._WIP_load_time_last: Timedelta = Timedelta()
        self.WIP_load_num_jobs: int = 0
        self._WIP_load_num_jobs_last: int = 0
        self.WIP_load_time_remaining: Timedelta = Timedelta()
        self.contents_WIP_load_time: Timedelta = Timedelta()
        self.contents_WIP_load_time_remaining: Timedelta = Timedelta()
        self.contents_WIP_load_num_jobs: int = 0
        self.num_inputs: int = 0
        self.num_outputs: int = 0
        self.WIP_inflow: Timedelta = Timedelta()
        self.WIP_outflow: Timedelta = Timedelta()

    @property
    def wei_avg_WIP_level_time(self) -> Timedelta | None:
        return self._wei_avg_WIP_level_time

    @property
    def wei_avg_WIP_level_num(self) -> float | None:
        return self._wei_avg_WIP_level_num

    @property
    def WIP_time_db(self) -> DataFrame:
        return self._WIP_time_db

    @property
    def WIP_num_db(self) -> DataFrame:
        return self._WIP_num_db

    def _calc_contents_WIP(self) -> None:
        WIP_num: int = 0
        self._current_op_remaining()
        current_op = self.target_object.current_op
        if current_op is not None:
            WIP_num = 1

        self.contents_WIP_load_time = self.time_op_processed + self.time_op_remaining
        self.contents_WIP_load_time_remaining = self.time_op_remaining
        self.contents_WIP_load_num_jobs = WIP_num

        loggers.monitors.debug(
            (
                '[Monitor] [Target Object >>%s<<] Contents WIP Load Time: %s, '
                'Contents WIP Load Time remaining: %s'
            ),
            self.target_object,
            self.contents_WIP_load_time,
            self.contents_WIP_load_time_remaining,
        )

    def _calc_normal_WIP(self) -> None:
        self._calc_contents_WIP()
        log_q = self.target_object.logical_queue
        relevant_station_group = cast(
            'StationGroup', self.target_object.supersystems_as_tuple()[0]
        )
        q_WIP_load_time, q_WIP_load_time_remaining, q_WIP_load_num_jobs = (
            log_q.calc_contents_WIP_filter(filter_station_group=relevant_station_group)
        )

        self.WIP_load_time = q_WIP_load_time + self.contents_WIP_load_time
        self.WIP_load_time_remaining = (
            q_WIP_load_time_remaining + self.contents_WIP_load_time_remaining
        )
        self.WIP_load_num_jobs = q_WIP_load_num_jobs + self.contents_WIP_load_num_jobs

        loggers.monitors.debug(
            (
                '[Monitor] [Target Object >>%s<<] Normal WIP Load Time: %s, '
                'Normal WIP Load Time remaining: %s, Normal WIP num jobs: %d'
            ),
            self.target_object,
            self.WIP_load_time,
            self.WIP_load_time_remaining,
            self.WIP_load_num_jobs,
        )

    def _current_op_remaining(self) -> None:
        target_object = self.target_object
        current_op = target_object.current_op
        if current_op is None:
            self.time_op_processed = Timedelta()
            self.time_op_remaining = Timedelta()
        elif current_op.time_actual_starting is None:
            self.time_op_processed = Timedelta()
            self.time_op_remaining = current_op.order_time
        else:
            current_sim_time = self.env.t_as_dt()
            time_op_processed = current_sim_time - current_op.time_actual_starting
            if time_op_processed < Timedelta():
                raise ValueError(
                    f'Already processed time of >>{current_op}<< must not be negative.'
                )
            self.time_op_processed = time_op_processed
            self.time_op_remaining = current_op.order_time - time_op_processed

            loggers.monitors.debug(
                '[Monitor] [OP >>%s<<] total order time: %s, already processed: %s, remaining: %s at ENV-TIME: %s',
                current_op,
                current_op.order_time,
                self.time_op_processed,
                self.time_op_remaining,
                current_sim_time,
            )

    @override
    def calc_KPI(self) -> None:
        """calculates different KPIs at any point in time"""
        super().calc_KPI()
        # contents WIP calculation implicitly called by normal WIP calculation
        self._calc_normal_WIP()
        self._track_WIP_level()
        # utilisation
        if self.time_non_helpers.total_seconds() > 0:
            self.utilisation = self.time_utilisation / self.time_non_helpers
            loggers.monitors.debug(
                'Utilisation of %s: %.3f at %s',
                self.target_object,
                self.utilisation,
                self.env.t_as_dt(),
            )

    def _track_WIP_level(
        self,
        is_finalise: bool = False,
    ) -> None:
        """adds an entry to the fill level database"""
        # only calculate duration if level changes
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        self._calc_normal_WIP()

        if (self._WIP_load_time_last != self.WIP_load_time) or is_finalise:
            # if updates occur at an already set time, just update the level
            if self._WIP_time_starting_time == current_time:
                self._WIP_time_db.iat[-1, 2] = self.WIP_load_time
                self._WIP_load_time_last = self.WIP_load_time
            # else new entry
            else:
                duration = current_time - self._WIP_time_starting_time
                temp1: Series = pd.Series(
                    index=['sim_time', 'duration', 'level'],
                    data=[current_time, duration, self.WIP_load_time],
                )
                temp2: DataFrame = temp1.to_frame().T.astype(self._WIP_time_db_types)
                self._WIP_time_db = pd.concat([self._WIP_time_db, temp2], ignore_index=True)
                self._WIP_load_time_last = self.WIP_load_time
                self._WIP_time_starting_time = current_time

        if (self._WIP_load_num_jobs_last != self.WIP_load_num_jobs) or is_finalise:
            # if updates occur at an already set time, just update the level
            if self._WIP_num_starting_time == current_time:
                self._WIP_num_db.iat[-1, 2] = self.WIP_load_num_jobs
                self._WIP_load_num_jobs_last = self.WIP_load_num_jobs
            # else new entry
            else:
                duration = current_time - self._WIP_num_starting_time
                temp1: Series = pd.Series(
                    index=['sim_time', 'duration', 'level'],
                    data=[current_time, duration, self.WIP_load_num_jobs],
                )
                temp2: DataFrame = temp1.to_frame().T.astype(self._WIP_num_db_types)
                self._WIP_num_db = pd.concat([self._WIP_num_db, temp2], ignore_index=True)
                self._WIP_load_num_jobs_last = self.WIP_load_num_jobs
                self._WIP_num_starting_time = current_time

    def _change_WIP_num(
        self,
        remove: bool,
    ) -> None:
        if remove:
            self.num_outputs += 1
        else:
            self.num_inputs += 1

    def _change_WIP(
        self,
        job: Job,
        remove: bool,
    ) -> None:
        # removing WIP
        if remove:
            # next operation of the job already assigned
            if job.last_order_time is None:
                raise ValueError(f'Last order time of job {job} is not set.')
            self.WIP_load_time -= job.last_order_time
            self.WIP_load_num_jobs -= 1
            self.WIP_outflow += job.last_order_time
            # self.num_outputs += 1
        else:
            if job.current_order_time is None:
                raise ValueError(f'Current order time of job {job} is not set.')
            self.WIP_load_time += job.current_order_time
            self.WIP_load_num_jobs += 1
            self.WIP_inflow += job.current_order_time
            # self.num_inputs += 1

        self._change_WIP_num(remove=remove)
        self._track_WIP_level()

    @override
    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # execute parent class function
        super().finalise_stats()

        # finalise WIP level tracking
        self._track_WIP_level(is_finalise=True)

        # post-process WIP time level databases
        self._WIP_time_db['level'] = self._WIP_time_db['level'].shift(
            periods=1, fill_value=Timedelta()
        )
        self._WIP_time_db = self._WIP_time_db.loc[
            self._WIP_time_db['duration'] > Timedelta(), :
        ].copy()
        self._WIP_time_db = self._WIP_time_db.reset_index(drop=True)

        # weighted average WIP time level
        temp1: DataFrame = self._WIP_time_db.copy()
        temp1['level_seconds'] = temp1['level'].apply(func=lambda x: x.total_seconds())
        temp1['duration_seconds'] = temp1['duration'].apply(func=lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level_seconds']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        wei_avg_time_sec: float = sums['mul'] / (sums['duration_seconds'] + EPSILON)
        self._wei_avg_WIP_level_time = Timedelta(seconds=wei_avg_time_sec)

        # post-process WIP num level databases
        self._WIP_num_db['level'] = self._WIP_num_db['level'].shift(
            periods=1, fill_value=Timedelta()
        )
        self._WIP_num_db = self._WIP_num_db.loc[
            self._WIP_num_db['duration'] > Timedelta(), :
        ].copy()
        self._WIP_num_db = self._WIP_num_db.reset_index(drop=True)
        # weighted average WIP num level
        temp1: DataFrame = self._WIP_num_db.copy()
        temp1['duration_seconds'] = temp1['duration'].apply(func=lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        self._wei_avg_WIP_level_num = sums['mul'] / (sums['duration_seconds'] + EPSILON)

    ### ANALYSE AND CHARTS ###
    def draw_WIP_level(
        self,
        use_num_jobs_metric: bool = False,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'fill_level',
        time_unit_load_time: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
    ) -> PlotlyFigure:
        """
        method to draw and display the fill level expansion of the corresponding buffer
        """
        if self._wei_avg_WIP_level_num is None:
            raise ValueError('Weighted average WIP level is not set.')
        if self._wei_avg_WIP_level_time is None:
            raise ValueError('Weighted average WIP level is not set.')
        # add starting point to start chart at t = init time
        title: str
        yaxis: str
        avg_WIP_level: float
        last_WIP_level: float
        if use_num_jobs_metric:
            data = self._WIP_num_db.copy()
            title = f'WIP Level Num Jobs of {self._target_object}'
            yaxis = 'WIP Level Number of Jobs [-]'
            avg_WIP_level = self._wei_avg_WIP_level_num
            # last_WIP_level = self.WIP_load_time
            last_WIP_level = self.WIP_load_num_jobs
        else:
            data = self._WIP_time_db.copy()
            # change WIP load time from Timedelta to any time unit possible --> float
            # Plotly can not handle Timedelta objects properly, only Datetimes
            calc_td = pyf_dt.timedelta_from_val(val=1.0, time_unit=time_unit_load_time)
            data['level'] = data['level'] / calc_td  # type: ignore
            title = f'WIP Level Time of {self._target_object}'
            yaxis = 'WIP Level Time [time units]'
            avg_WIP_level = cast(float, self._wei_avg_WIP_level_time / calc_td)
            last_WIP_level = cast(float, self.WIP_load_time / calc_td)
        f_val1 = cast(Datetime, data.at[0, 'sim_time'] - data.at[0, 'duration'])
        f_val2 = Timedelta()
        f_val3 = cast(float, data.at[0, 'level'])
        first_entry = pd.DataFrame(columns=data.columns, data=[[f_val1, f_val2, f_val3]])
        l_val1 = cast(Datetime, data.iat[-1, 0])
        l_val2 = Timedelta()
        l_val3 = last_WIP_level
        last_entry = pd.DataFrame(columns=data.columns, data=[[l_val1, l_val2, l_val3]])
        temp1 = pd.concat([first_entry, data, last_entry], ignore_index=True)

        fig = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape='vh')
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title=title)
        fig.update_yaxes(title=dict({'text': yaxis}))
        fig.update_xaxes(title=dict({'text': 'time'}))
        # weighted average WIP level
        fig.add_hline(y=avg_WIP_level, line_width=3, line_dash='dot', line_color='orange')

        fig.show()

        file_name = file_name + f'_{self}'

        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)

        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)

        return fig
