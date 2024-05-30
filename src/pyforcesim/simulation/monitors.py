from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, cast

import pandas as pd
import plotly.express as px
from pandas import DataFrame, Series

from pyforcesim import loggers
from pyforcesim.constants import INF
from pyforcesim.datetime import DTManager
from pyforcesim.types import PlotlyFigure

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        InfrastructureObject,
        Job,
        Operation,
        SimulationEnvironment,
        StorageLike,
    )

_dt_mgr = DTManager()


class Monitor:
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: InfrastructureObject | Job | Operation,
        init_state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'WAITING',
            'PROCESSING',
            'SETUP',
            'BLOCKED',
            'FAILED',
            'PAUSED',
        ),
    ) -> None:
        """
        Class to monitor associated objects (load and resource)
        """
        # initialise parent class if available
        # super().__init__(**kwargs)

        # [REGISTRATION]
        self._env = env
        self._target_object = obj

        # [STATE] state parameters
        # always add states 'INIT', 'FINISH', 'TEMP' for control flow
        states_possible = set(possible_states)
        if 'INIT' not in states_possible:
            states_possible.add('INIT')
        if 'FINISH' not in states_possible:
            states_possible.add('FINISH')
        if 'TEMP' not in states_possible:
            states_possible.add('TEMP')
        # all possible/allowed states
        self.states_possible: frozenset[str] = frozenset(states_possible)

        # check integrity of the given state
        if init_state in self.states_possible:
            self.state_current: str = init_state
        else:
            raise ValueError(
                (
                    f'The state {init_state} is not allowed. '
                    f'Must be one of {self.states_possible}'
                )
            )

        # boolean indicator if a state is set
        self.state_status: dict[str, bool] = {}
        # time counter for each state
        # self.state_times: dict[str, float] = {}
        self.state_times: dict[str, Timedelta] = {}
        # starting time variable indicating when the last state assignment took place
        # self.state_starting_time: float = self._env.t()
        self.state_starting_time: Datetime = self._env.t_as_dt()

        for state in self.states_possible:
            # init state time dictionary
            # self.state_times[state] = 0.
            self.state_times[state] = Timedelta()
            # init state is set to True
            if state == self.state_current:
                self.state_status[state] = True
            else:
                self.state_status[state] = False

        # DataFrame to further analyse state durations
        self.state_durations: DataFrame | None = None

        # availability indicator
        self._availability_states: set[str] = set(
            [
                'WAITING',
            ]
        )
        if self.state_current in self._availability_states:
            self.is_available: bool = True
        else:
            self.is_available: bool = False

        # additional 'TEMP' state information
        # indicator if state was 'TEMP'
        self._is_temp: bool = False
        # state before 'TEMP' was set
        self._state_before_temp: str = self.state_current
        # time components
        self.time_active: float = 0.0
        # self.time_active: Timedelta = Timedelta()

        # time handling
        # self._dt_parser: DTParser = DTParser()
        # if isinstance(self.target_object, Operation):
        loggers.monitors.debug('Monitor states: %s', self.states_possible)
        loggers.monitors.debug('Monitor state times: %s', self.state_times)

    def __repr__(self) -> str:
        return f'Monitor instance of {self.target_object}'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def target_object(self) -> InfrastructureObject | Job | Operation:
        return self._target_object

    def get_current_state(self) -> str:
        """get the current state of the associated resource"""
        return self.state_current

    def set_state(
        self,
        state: str,
    ) -> None:
        """
        function to set the object in the given state
        state: name of the state in which the object should be placed, must be part \
            of the object's possible states
        """
        # eliminate lower-case letters
        target_state = state.upper()

        # check if state is allowed
        if target_state not in self.states_possible:
            raise ValueError(
                (
                    f'The state {target_state} is not allowed. '
                    f'Must be one of {self.states_possible}'
                )
            )

        # check if state is already set
        if self.state_status[target_state] and target_state != 'TEMP':
            loggers.monitors.info(
                'Tried to set state of %s to >>%s<<, but this state was already set.'
                ' State of object was not changed.',
                self.target_object,
                target_state,
            )
        # check if the 'TEMP' state was already set, this should never happen
        # if it happens raise an error to catch wrong behaviour
        elif self.state_status[target_state] and target_state == 'TEMP':
            raise RuntimeError(
                (
                    f'Tried to set state of {self._target_object} to >>TEMP<<, '
                    f'but this state was already set.'
                )
            )

        # calculate time for which the object was in the current state before changing it
        current_state_start = self.state_starting_time
        # current_time = self._env.now()
        current_time = self._env.t_as_dt()
        current_state_duration: Timedelta = current_time - current_state_start
        # add time to the time counter for the current state
        current_state = self.state_current
        self.state_times[current_state] += current_state_duration

        # check if 'TEMP' state shall be set
        if target_state == 'TEMP':
            # set 'TEMP' state indicator to true
            self._is_temp = True
            # save current state for the state reset
            self._state_before_temp = current_state

        # set old state to False and new state to True
        self.state_status[current_state] = False
        self.state_status[target_state] = True
        # assign new state as current one
        self.state_current = target_state
        # set state starting time to current time
        self.state_starting_time = current_time
        # availability
        if self.state_current in self._availability_states:
            self.is_available: bool = True
        elif self.state_current == 'TEMP':
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
        if not self._is_temp:
            raise RuntimeError(
                (
                    f'Tried to reset {self._target_object} from >>TEMP<< state but '
                    f'the current state is >>{self.state_current}<<'
                )
            )
        else:
            self._is_temp = False
            self.set_state(state=self._state_before_temp)

    def calc_KPI(
        self,
        is_finalise: bool = False,
    ) -> None:
        """calculates different KPIs at any point in time"""

        # state durations for analysis
        if not is_finalise:
            self.state_durations = self.state_durations_as_df()

        # [TOTAL ACTIVE TIME]
        if self.state_durations is None:
            raise RuntimeError('State durations are not available. Can not calculate KPIs.')
        self.time_active = self.state_durations.loc[:, 'abs [seconds]'].sum()

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
        temp2['rel [%]'] = temp2['abs [seconds]'] / temp2.sum(axis=0)['abs [seconds]'] * 100
        temp2 = temp2.drop(labels=['INIT', 'FINISH', 'TEMP'], axis=0)
        temp2 = temp2.sort_index(axis=0, ascending=True, kind='stable')
        state_durations_df = temp2.copy()

        return state_durations_df

    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""

        # assign state duration table
        # !! REWORK with calc_KPI
        self.state_durations = self.state_durations_as_df()

        # calculate KPIs
        self.calc_KPI(is_finalise=True)

    ### ANALYSE AND CHARTS ###
    def draw_state_bar_chart(
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'state_distribution_bar',
        time_unit: str = 'hours',
    ) -> PlotlyFigure:
        """draws the collected state times of the object as bar chart"""
        data = pd.DataFrame.from_dict(
            data=self.state_times, orient='index', columns=['total time']
        )
        data.index = data.index.rename('state')
        # change time from Timedelta to any time unit possible --> float
        # Plotly can not handle Timedelta objects properly, only Datetimes
        calc_td = _dt_mgr.timedelta_from_val(val=1.0, time_unit=time_unit)
        calc_col: str = f'total time [{time_unit}]'
        data[calc_col] = data['total time'] / calc_td  # type: ignore
        data = data.sort_index(axis=0, kind='stable')

        fig: PlotlyFigure = px.bar(data, y=calc_col, text_auto='.2f')  # type: ignore wrong type hint in Plotly
        fig.update_layout(
            title=f'State Time Distribution of {self._target_object}', showlegend=False
        )
        fig.update_yaxes(title=dict({'text': calc_col}))

        fig.show()

        file_name = file_name + f'_{self}'

        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)

        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)

        return fig

    def draw_state_pie_chart(
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'state_distribution_pie',
        time_unit: str = 'hours',
    ) -> PlotlyFigure:
        """draws the collected state times of the object as bar chart"""
        data = pd.DataFrame.from_dict(
            data=self.state_times, orient='index', columns=['total time']
        )
        data.index = data.index.rename('state')
        # change time from Timedelta to any time unit possible --> float
        # Plotly can not handle Timedelta objects properly, only Datetimes
        calc_td = _dt_mgr.timedelta_from_val(val=1.0, time_unit=time_unit)
        calc_col: str = f'total time [{time_unit}]'
        data[calc_col] = data['total time'] / calc_td  # type: ignore
        data = data.sort_index(axis=0, kind='stable')
        data = data.loc[data[calc_col] > 0.0, :]

        fig: PlotlyFigure = px.pie(data, values=calc_col, names=data.index)
        fig.update_layout(title=f'State Time Distribution of {self._target_object}')

        fig.show()

        file_name = file_name + f'_{self}'

        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)

        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)

        return fig


class StorageMonitor(Monitor):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: StorageLike,
        init_state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'FULL',
            'EMPTY',
            'INTERMEDIATE',
            'FAILED',
            'PAUSED',
        ),
    ) -> None:
        # initialise parent class
        super().__init__(
            env=env,
            obj=obj,
            init_state=init_state,
            possible_states=possible_states,
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
        # self._fill_level_starting_time: float = self.env.now()
        self._fill_level_starting_time: Datetime = self.env.t_as_dt()
        self._wei_avg_fill_level: float | None = None

        # overwrite
        self._target_object = obj

    @property
    def target_object(self) -> StorageLike:
        return self._target_object

    @property
    def wei_avg_fill_level(self) -> float | None:
        return self._wei_avg_fill_level

    @property
    def level_db(self) -> DataFrame:
        return self._level_db

    def set_state(
        self,
        state: str,
    ) -> None:
        """additional level tracking functionality"""
        super().set_state(state=state)

        is_finalise: bool = False
        if self.state_current == 'FINISH':
            is_finalise: bool = True
        self._track_fill_level(is_finalise=is_finalise)

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
            len(self.target_object),
            self._current_fill_level,
        )
        # if ((self._current_fill_level != len(self)) and (duration > 0.0)) or is_finalise:
        if self._current_fill_level != self._target_object.fill_level or is_finalise:
            temp1: Series = pd.Series(
                index=['sim_time', 'duration', 'level'],
                data=[current_time, duration, self._current_fill_level],
            )
            temp2: DataFrame = temp1.to_frame().T.astype(self._level_db_types)
            self._level_db = pd.concat([self._level_db, temp2], ignore_index=True)
            self._current_fill_level = self._target_object.fill_level
            self._fill_level_starting_time = current_time

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
        self._wei_avg_fill_level = cast(float, sums['mul'] / sums['duration_seconds'])

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

        fig: PlotlyFigure = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape='vh')
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


class InfStructMonitor(Monitor):
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: InfrastructureObject,
        init_state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'WAITING',
            'PROCESSING',
            'SETUP',
            'BLOCKED',
            'FAILED',
            'PAUSED',
        ),
    ) -> None:
        # initialise parent class
        super().__init__(
            env=env,
            obj=obj,
            init_state=init_state,
            possible_states=possible_states,
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

        # logistic objective values
        # self.WIP_load_time: float = 0.
        # self._WIP_load_time_last: float = 0.
        self.WIP_load_time: Timedelta = Timedelta()
        self._WIP_load_time_last: Timedelta = Timedelta()
        self.WIP_load_num_jobs: int = 0
        self._WIP_load_num_jobs_last: int = 0

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

    def _track_WIP_level(
        self,
        is_finalise: bool = False,
    ) -> None:
        """adds an entry to the fill level database"""
        # only calculate duration if level changes
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()

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

    def calc_KPI(
        self,
        is_finalise: bool = False,
    ) -> None:
        super().calc_KPI()

        # [OCCUPATION]
        # properties which count as occupied
        # paused counts in because pausing the processing station is an external factor
        util_props = ['PROCESSING', 'SETUP', 'PAUSED']
        self.time_occupied = self.state_durations.loc[util_props, 'abs [seconds]'].sum()  # type: ignore

        # [UTILISATION]
        # avoid division by 0
        if self.time_active > 0.0:
            self.utilisation: float = self.time_occupied / self.time_active

    def change_WIP(
        self,
        job: 'Job',
        remove: bool,
    ) -> None:
        # removing WIP
        if remove:
            # next operation of the job already assigned
            # self.WIP_load_time -= job.last_proc_time
            if job.last_order_time is None:
                raise ValueError(f'Last order time of job {job} is not set.')
            self.WIP_load_time -= job.last_order_time
            self.WIP_load_num_jobs -= 1
        else:
            # self.WIP_load_time += job.current_proc_time
            if job.current_order_time is None:
                raise ValueError(f'Current order time of job {job} is not set.')
            self.WIP_load_time += job.current_order_time
            self.WIP_load_num_jobs += 1

        self._track_WIP_level()

    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # execute parent class function
        super().finalise_stats()

        # finalise WIP level tracking
        self._track_WIP_level(is_finalise=True)

        # post-process WIP time level databases
        # print(f'I AM {self}')
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
        wei_avg_time_sec: float = sums['mul'] / sums['duration_seconds']
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
        self._wei_avg_WIP_level_num = sums['mul'] / sums['duration_seconds']

    ### ANALYSE AND CHARTS ###
    def draw_WIP_level(
        self,
        use_num_jobs_metric: bool = False,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'fill_level',
        time_unit_load_time: str = 'hours',
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
            calc_td = _dt_mgr.timedelta_from_val(val=1.0, time_unit=time_unit_load_time)
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

        fig: PlotlyFigure = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape='vh')
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
