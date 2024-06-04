"""Module with several building blocks for simulation environments"""

from __future__ import annotations

import multiprocessing as mp
import random
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable, Iterator, Sequence
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from functools import lru_cache
from operator import attrgetter
from typing import (
    Any,
    Final,
    Literal,
    Self,
    cast,
    overload,
)

import pandas as pd
import plotly.express as px
import plotly.io
import salabim
from pandas import DataFrame
from websocket import create_connection

from pyforcesim import loggers
from pyforcesim.common import enum_str_values_as_frzset, flatten
from pyforcesim.constants import (
    INF,
    POLICIES_ALLOC,
    POLICIES_SEQ,
    SimStatesCommon,
    SimStatesStorage,
    SimSystemTypes,
)
from pyforcesim.dashboard.dashboard import (
    WS_URL,
    start_dashboard,
)
from pyforcesim.dashboard.websocket_server import start_websocket_server
from pyforcesim.datetime import (
    DEFAULT_DATETIME,
    TIMEZONE_UTC,
    DTManager,
    adjust_db_dates_local_tz,
)
from pyforcesim.errors import AssociationError
from pyforcesim.rl.agents import Agent, AllocationAgent
from pyforcesim.simulation import monitors
from pyforcesim.simulation.base_components import (
    SimulationComponent,
    StorageComponent,
)
from pyforcesim.simulation.loads import (
    OrderTime,
    RandomJobGenerator,
)
from pyforcesim.simulation.policies import (
    AllocationPolicy,
    GeneralPolicy,
    SequencingPolicy,
)
from pyforcesim.types import (
    AgentTasks,
    CustomID,
    Infinite,
    LoadID,
    PlotlyFigure,
    SystemID,
)

# ** constants
# definition of routing system level
EXEC_SYSTEM_TYPE: Final = SimSystemTypes.PRODUCTION_AREA
# time after a store request is failed
FAIL_DELAY: Final[float] = 20.0

# ** utilities
# UTILITIES: Datetime Manager
_dt_mgr = DTManager()


def filter_processing_stations(
    infstruct_obj_collection: Iterable[InfrastructureObject],
) -> list[ProcessingStation]:
    """Filters an iterable with InfrastructureObjects for ProcessingStations

    Parameters
    ----------
    infstruct_obj_collection : Iterable[InfrastructureObject]
        collection of InfrastrcutureObjects

    Returns
    -------
    list[ProcessingStation]
        list of ProcessingStations for the given collection
    """

    return [x for x in infstruct_obj_collection if isinstance(x, ProcessingStation)]


# ** environment


class SimulationEnvironment(salabim.Environment):
    def __init__(
        self,
        time_unit: str = 'seconds',
        starting_datetime: Datetime | None = None,
        debug_dashboard: bool = False,
        **kwargs,
    ) -> None:
        """Simulation Environment:
        contains all simulation entities and manages the simulation run
        every entity is registered in or associated with the environment

        Parameters
        ----------
        time_unit : str, optional
            time unit internally used to represent intervals, by default 'seconds'
        starting_datetime : Datetime | None, optional
            starting date and time (t=0 for environment), by default None
        debug_dashboard : bool, optional
            using a debug dashboard implemented in Dash for testing purposes,
            by default False
        """
        # time units
        self.time_unit = time_unit
        # if starting datetime not provided use current time
        if starting_datetime is None:
            starting_datetime = _dt_mgr.current_time_tz(cut_microseconds=True)
        else:
            _dt_mgr.validate_dt_UTC(starting_datetime)
            # remove microseconds, such accuracy not needed
            starting_datetime = _dt_mgr.cut_dt_microseconds(dt=starting_datetime)
        self.starting_datetime = starting_datetime

        super().__init__(
            trace=False,
            time_unit=self.time_unit,
            datetime0=self.starting_datetime,
            yieldless=False,
            **kwargs,
        )

        # [RESOURCE] infrastructure manager
        self._infstruct_mgr = InfrastructureManager(env=self)
        loggers.pyf_env.info(
            'Successfully registered Infrastructure Manager in Env >>%s<<', self.name()
        )
        # [LOAD] job dispatcher
        self._dispatcher: Dispatcher = Dispatcher(env=self)
        loggers.pyf_env.info('Successfully registered Dispatcher in Env >>%s<<', self.name())
        # transient condition
        # state allows direct waiting for flag changes
        self.is_transient_cond: bool = True
        self.transient_cond_state = salabim.State('trans_cond', env=self)

        # ** debug dashboard
        self.debug_dashboard = debug_dashboard
        self.servers_connected: bool = False
        if self.debug_dashboard:
            self.ws_server_process = mp.Process(target=start_websocket_server)
            self.dashboard_server_process = mp.Process(target=start_dashboard)

    def t_as_dt(self) -> Datetime:
        """return current simulation time as Datetime object

        Returns
        -------
        Datetime
            simulation time in current time unit as Datetime object
        """
        return self.t_to_datetime(t=self.t())

    def td_to_simtime(
        self,
        timedelta: Timedelta,
    ) -> float:
        """transform Timedelta to simulation time"""
        return self.timedelta_to_duration(timedelta=timedelta)

    @property
    def infstruct_mgr(self) -> InfrastructureManager:
        """obtain the current registered Infrastructure Manager instance of the environment"""
        if self._infstruct_mgr is None:
            raise ValueError('No Infrastructure Manager instance registered.')
        else:
            return self._infstruct_mgr

    @property
    def dispatcher(self) -> Dispatcher:
        """obtain the current registered Dispatcher instance of the environment"""
        if self._dispatcher is None:
            raise ValueError('No Dipsatcher instance registered.')
        else:
            return self._dispatcher

    # ?? adding to Dispatcher class?
    def check_feasible_agent_alloc(
        self,
        target_station: ProcessingStation,
        op: Operation,
    ) -> bool:
        """
        method which checks for feasibility of agent allocation decisions
        returning True if feasible, False otherwise
        """
        # check if operation has station group identifier (SGI) (SystemID)
        op_SGI = op.target_station_group_identifier

        # no station group assigned, chosen station is automatically feasible
        if op_SGI is None:
            return True
        else:
            # lookup SGIs of the target station's station groups
            # target_SGIs = target_station.supersystems_custom_ids
            target_SGIs = target_station.supersystems_ids

        if op_SGI in target_SGIs:
            # operation SGI in associated station group IDs found,
            # target station is feasible for given operation
            return True
        else:
            return False

    def check_integrity(self) -> None:
        """
        method to evaluate if certain criteria for the simulation run are satisfied
        checks for:
        - registered dispatcher (min: 1, max: 1)
        - registered sink (min: 1, max: INF)
        """
        # registration of an Infrastructure Manager
        # if not self._infstruct_mgr_registered:
        #     raise ValueError('No Infrastructure Manager instance registered.')
        # registration of a Dispatcher
        # if not self._dispatcher_registered:
        #     raise ValueError('No Dispatcher instance registered.')
        # registration of sinks
        if not self._infstruct_mgr.sink_registered:
            raise ValueError('No Sink instance registered.')
        # check if all subsystems are associated to supersystems
        elif not self._infstruct_mgr.verify_system_association():
            raise AssociationError('Non-associated subsystems detected!')

        loggers.pyf_env.info(
            'Integrity check for Environment >>%s<< successful.', self.name()
        )

    def initialise(self) -> None:
        # infrastructure manager instance
        self._infstruct_mgr.initialise()

        # dispatcher instance
        self._dispatcher.initialise()

        # establish websocket connection
        if self.debug_dashboard and not self.servers_connected:
            # start websocket server
            loggers.pyf_env.info('Starting websocket server...')
            self.ws_server_process.start()
            # start dashboard server
            loggers.pyf_env.info('Starting dashboard server...')
            self.dashboard_server_process.start()
            # establish websocket connection used for streaming of updates
            loggers.pyf_env.info('Establish websocket connection...')
            self.ws_con = create_connection(WS_URL)
            # set internal flag indicating that servers are started
            self.servers_connected = True

        loggers.pyf_env.info('Initialisation for Environment >>%s<< successful.', self.name())

    def finalise(self) -> None:
        """
        Function which should be executed at the end of the simulation.
        Can be used for finalising data collection, other related tasks or
        further processing pipelines
        """
        # infrastructure manager instance
        self._infstruct_mgr.finalise()
        # dispatcher instance
        self._dispatcher.finalise()
        # close WS connection
        if self.debug_dashboard and self.servers_connected:
            # close websocket connection
            loggers.pyf_env.info('Closing websocket connection...')
            self.ws_con.close()
            # stop websocket server
            loggers.pyf_env.info('Shutting down websocket server...')
            self.ws_server_process.terminate()
            # stop dashboard server
            loggers.pyf_env.info('Shutting down dasboard server...')
            self.dashboard_server_process.terminate()
            # reset internal flag indicating that servers are started
            self.servers_connected = False

    def dashboard_update(self) -> None:
        # infrastructure manager instance
        self._infstruct_mgr.dashboard_update()

        # dispatcher instance
        self._dispatcher.dashboard_update()


# ** environment management


class InfrastructureManager:
    def __init__(
        self,
        env: SimulationEnvironment,
    ) -> None:
        # COMMON
        self._env = env
        # subsystem types
        # self._system_types: set[str] = set(
        #     [
        #         'ProductionArea',
        #         'StationGroup',
        #         'Resource',
        #     ]
        # )
        # self._system_types: frozenset[str] = frozenset((val.value for val in SimSystemTypes))
        self._system_types = enum_str_values_as_frzset(SimSystemTypes)

        # PRODUCTION AREAS database as simple Pandas DataFrame
        self._prod_area_prop: dict[str, type] = {
            'prod_area_id': int,
            'custom_id': object,
            'name': str,
            'prod_area': object,
            'containing_proc_stations': bool,
        }
        self._prod_area_db: DataFrame = pd.DataFrame(
            columns=list(self._prod_area_prop.keys())
        )
        self._prod_area_db = self._prod_area_db.astype(self._prod_area_prop)
        self._prod_area_db = self._prod_area_db.set_index('prod_area_id')
        self._prod_area_lookup_props: set[str] = set(['prod_area_id', 'custom_id', 'name'])
        # [PRODUCTION AREAS] identifiers
        self._prod_area_counter = SystemID(0)
        self._prod_area_custom_identifiers: set[CustomID] = set()

        # [STATION GROUPS] database as simple Pandas DataFrame
        self._station_group_prop: dict[str, type | pd.Int64Dtype] = {
            'station_group_id': int,
            'custom_id': object,
            'name': str,
            'station_group': object,
            'prod_area_id': pd.Int64Dtype(),
            'containing_proc_stations': bool,
        }
        self._station_group_db: DataFrame = pd.DataFrame(
            columns=list(self._station_group_prop.keys())
        )
        self._station_group_db = self._station_group_db.astype(self._station_group_prop)
        self._station_group_db = self._station_group_db.set_index('station_group_id')
        self._station_group_lookup_props: set[str] = set(
            ['station_group_id', 'custom_id', 'name']
        )
        # [STATION GROUPS] identifiers
        self._station_group_counter = SystemID(0)
        self._station_groups_custom_identifiers: set[CustomID] = set()

        # [RESOURCES] database as simple Pandas DataFrame
        self._infstruct_prop: dict[str, type | pd.Int64Dtype] = {
            'res_id': int,
            'custom_id': object,
            'resource': object,
            'name': str,
            'res_type': str,
            'state': str,
            'station_group_id': pd.Int64Dtype(),
        }
        self._res_db: DataFrame = pd.DataFrame(columns=list(self._infstruct_prop.keys()))
        self._res_db = self._res_db.astype(self._infstruct_prop)
        self._res_db = self._res_db.set_index('res_id')
        self._res_lookup_props: set[str] = set(['res_id', 'custom_id', 'name'])
        # [RESOURCES] custom identifiers
        self._res_counter = SystemID(0)
        self._res_custom_identifiers: set[CustomID] = set()
        # [RESOURCES] sink: pool of sinks possible to allow multiple sinks in one environment
        # [PERHAPS CHANGED LATER]
        # currently only one sink out of the pool is chosen because jobs do not contain
        # information about a target sink
        self._sink_registered: bool = False
        self._sinks: list[Sink] = []

        # counter for processing stations (machines, assembly, etc.)
        self.num_proc_stations: int = 0

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    # [PRODUCTION AREAS]
    @property
    def prod_area_db(self) -> DataFrame:
        return self._prod_area_db

    # [STATION GROUPS]
    @property
    def station_group_db(self) -> DataFrame:
        return self._station_group_db

    def verify_system_association(self) -> bool:
        """checks if there are any registered, but non-associated
        subsystems for each subsystem type

        Returns
        -------
        bool
            indicator if all systems are associated (True) or not (False)
        """
        # check all subsystem types with reference to supersystems if there are
        # any open references (NA values as secondary key)
        relevant_subsystems = (SimSystemTypes.STATION_GROUP, SimSystemTypes.RESOURCE)

        for system_type in relevant_subsystems:
            match system_type:
                case SimSystemTypes.STATION_GROUP:
                    target_db = self._station_group_db
                    secondary_key: str = 'prod_area_id'
                case SimSystemTypes.RESOURCE:
                    target_db = self._res_db
                    secondary_key: str = 'station_group_id'
            # check if there are any NA values as secondary key
            check_val = target_db[secondary_key].isna().any()
            if check_val:
                # there are NA values
                loggers.infstrct.error(
                    (
                        'There are non-associated systems for '
                        'system type >>%s<<. '
                        'Please check these systems and add them to a '
                        'corresponding supersystem.'
                    ),
                    system_type,
                )
                return False

        return True

    ####################################################################################
    ## REWORK TO WORK WITH DIFFERENT SUBSYSTEMS
    # only one register method by analogy with 'lookup_subsystem_info'
    # currently checking for existence and registration implemented,
    # split into different methods
    # one to check whether such a subsystem already exists
    # another one registers a new subsystem
    # if check positive: return subsystem by 'lookup_subsystem_info'
    ### REWORK TO MULTIPLE SUBSYSTEMS
    def _obtain_system_id(
        self,
        system_type: SimSystemTypes,
    ) -> SystemID:
        """Simple counter function for managing system IDs

        Returns
        -------
        SystemID
            unique system ID
        """
        if system_type not in self._system_types:
            raise ValueError(
                (
                    f'The subsystem type >>{system_type}<< is not allowed. '
                    f'Choose from {self._system_types}'
                )
            )

        system_id: SystemID
        match system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                system_id = self._prod_area_counter
                self._prod_area_counter += 1
            case SimSystemTypes.STATION_GROUP:
                system_id = self._station_group_counter
                self._station_group_counter += 1
            case SimSystemTypes.RESOURCE:
                system_id = self._res_counter
                self._res_counter += 1

        return system_id

    def register_subsystem(
        self,
        system_type: SimSystemTypes,
        obj: System,
        custom_identifier: CustomID,
        name: str | None,
        state: str | None = None,
    ) -> tuple[SystemID, str]:
        """
        registers an infrastructure object in the environment by assigning an unique id and 
        adding the object to the associated resources of the environment
        
        obj: env resource = instance of a subclass of InfrastructureObject
        custom_identifier: user defined identifier
        name: custom name of the object, \
            default: None
        returns:
            SystemID: assigned resource ID
            str: assigned resource's name
        """
        if system_type not in self._system_types:
            raise ValueError(
                (
                    f'The subsystem type >>{system_type}<< is not allowed. '
                    f'Choose from {self._system_types}'
                )
            )

        match system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                custom_identifiers = self._prod_area_custom_identifiers
            case SimSystemTypes.STATION_GROUP:
                custom_identifiers = self._station_groups_custom_identifiers
            case SimSystemTypes.RESOURCE:
                custom_identifiers = self._res_custom_identifiers
            case _:
                raise ValueError(f'Unknown subsystem type of {system_type}')

        # check if value already exists
        if custom_identifier in custom_identifiers:
            raise ValueError(
                (
                    f'The custom identifier {custom_identifier} provided '
                    f'for subsystem type {system_type} '
                    f'already exists, but has to be unique.'
                )
            )
        else:
            custom_identifiers.add(custom_identifier)

        # obtain system ID
        system_id = self._obtain_system_id(system_type=system_type)

        # [RESOURCES] resource related data
        # register sinks
        if isinstance(obj, Sink):
            if not self._sink_registered:
                self._sink_registered = True
            self._sinks.append(obj)
        # count number of machines
        if isinstance(obj, ProcessingStation):
            self.num_proc_stations += 1

        # custom name
        if name is None:
            name = f'{type(obj).__name__}_env_{system_id}'

        # new entry for corresponding database
        match system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                new_entry: DataFrame = pd.DataFrame(
                    {
                        'prod_area_id': [system_id],
                        'custom_id': [custom_identifier],
                        'name': [name],
                        'prod_area': [obj],
                        'containing_proc_stations': [obj.containing_proc_stations],
                    }
                )
                new_entry = new_entry.astype(self._prod_area_prop)
                new_entry = new_entry.set_index('prod_area_id')
                self._prod_area_db = pd.concat([self._prod_area_db, new_entry])
            case SimSystemTypes.STATION_GROUP:
                new_entry: DataFrame = pd.DataFrame(
                    {
                        'station_group_id': [system_id],
                        'custom_id': [custom_identifier],
                        'name': [name],
                        'station_group': [obj],
                        'prod_area_id': [None],
                        'containing_proc_stations': [obj.containing_proc_stations],
                    }
                )
                new_entry = new_entry.astype(self._station_group_prop)
                new_entry = new_entry.set_index('station_group_id')
                self._station_group_db = pd.concat([self._station_group_db, new_entry])
            case SimSystemTypes.RESOURCE:
                new_entry: DataFrame = pd.DataFrame(
                    {
                        'res_id': [system_id],
                        'custom_id': [custom_identifier],
                        'resource': [obj],
                        'name': [name],
                        'res_type': [obj.res_type],  # type: ignore
                        'state': [state],
                        'station_group_id': [None],
                    }
                )
                new_entry = new_entry.astype(self._infstruct_prop)
                new_entry = new_entry.set_index('res_id')
                self._res_db = pd.concat([self._res_db, new_entry])

        loggers.infstrct.info(
            'Successfully registered object with SystemID >>%s<< and name >>%s<<',
            system_id,
            name,
        )

        return system_id, name

    def register_system_association(
        self,
        supersystem: System,
        subsystem: System,
    ) -> None:
        """associate two system types with each other in the corresponding databases

        Parameters
        ----------
        supersystem : System
            system to which the subsystem is added
        subsystem : System
            system which is added to the supersystem and to whose database the entry is made
        """
        # target subsystem type -> identify appropriate database
        system_type = subsystem.system_type

        match system_type:
            case SimSystemTypes.STATION_GROUP:
                target_db = self._station_group_db
                target_property: str = 'prod_area_id'
            case SimSystemTypes.RESOURCE:
                target_db = self._res_db
                target_property: str = 'station_group_id'
        # system IDs
        supersystem_id = supersystem.system_id
        subsystem_id = subsystem.system_id
        # write supersystem ID to subsystem database entry
        target_db.at[subsystem_id, target_property] = supersystem_id

    def set_contain_proc_station(
        self,
        system: System,
    ) -> None:
        match system.system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                lookup_db = self._prod_area_db
            case SimSystemTypes.STATION_GROUP:
                lookup_db = self._station_group_db

        lookup_db.at[system.system_id, 'containing_proc_stations'] = True
        system.containing_proc_stations = True

        # iterate over supersystems
        for supersystem in system.supersystems.values():
            if not supersystem.containing_proc_stations:
                self.set_contain_proc_station(system=supersystem)

    def lookup_subsystem_info(
        self,
        system_type: SimSystemTypes,
        lookup_val: SystemID | CustomID,
        lookup_property: str | None = None,
        target_property: str | None = None,
    ) -> Any:
        """
        obtain a subsystem by its property and corresponding value
        properties: Subsystem ID, Custom ID, Name
        """
        if system_type not in self._system_types:
            raise ValueError(
                (
                    f'The subsystem type >>{system_type}<< is not allowed. '
                    f'Choose from {self._system_types}'
                )
            )

        id_prop: str
        match system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                allowed_lookup_props = self._prod_area_lookup_props
                lookup_db = self._prod_area_db
                if target_property is None:
                    target_property = 'prod_area'
                id_prop = 'prod_area_id'
            case SimSystemTypes.STATION_GROUP:
                allowed_lookup_props = self._station_group_lookup_props
                lookup_db = self._station_group_db
                if target_property is None:
                    target_property = 'station_group'
                id_prop = 'station_group_id'
            case SimSystemTypes.RESOURCE:
                allowed_lookup_props = self._res_lookup_props
                lookup_db = self._res_db
                if target_property is None:
                    target_property = 'resource'
                id_prop = 'res_id'

        # if no lookup property provided use ID
        if lookup_property is None:
            lookup_property = id_prop

        # allowed target properties
        allowed_target_props: set[str] = set(lookup_db.columns.to_list())
        # lookup property can not be part of the target properties
        if lookup_property in allowed_target_props:
            allowed_target_props.remove(lookup_property)

        # check if property is a filter criterion
        if lookup_property not in allowed_lookup_props:
            raise IndexError(
                (
                    f'Lookup Property >>{lookup_property}<< is not allowed for '
                    f'subsystem type {system_type}. Choose from '
                    f'{allowed_lookup_props}'
                )
            )
        # check if target property is allowed
        if target_property not in allowed_target_props:
            raise IndexError(
                (
                    f'Target Property >>{target_property}<< is not allowed for '
                    f'subsystem type {system_type}. Choose from {allowed_target_props}'
                )
            )
        # None type value can not be looked for
        if lookup_val is None:
            raise TypeError('The lookup value can not be of type >>None<<.')

        # filter resource database for prop-value pair
        if lookup_property == id_prop:
            # direct indexing for ID property: always unique, no need for duplicate check
            try:
                idx_res: Any = lookup_db.at[lookup_val, target_property]
                return idx_res
            except KeyError:
                raise IndexError(
                    (
                        f'There were no subsystems found for the '
                        f'lookup property >>{lookup_property}<< '
                        f'with the value >>{lookup_val}<<'
                    )
                )
        else:
            try:
                multi_res = lookup_db.loc[
                    lookup_db[lookup_property] == lookup_val, target_property
                ]
                # check for empty search result, at least one result necessary
                if len(multi_res) == 0:
                    raise IndexError(
                        (
                            f'There were no subsystems found for the lookup '
                            f'property >>{lookup_property}<< '
                            f'with the value >>{lookup_val}<<'
                        )
                    )
            except KeyError:
                raise IndexError(
                    (
                        f'There were no subsystems found for the '
                        f'lookup property >>{lookup_property}<< '
                        f'with the value >>{lookup_val}<<'
                    )
                )
            # check for multiple entries with same prop-value pair
            ########### PERHAPS CHANGE NECESSARY
            ### multiple entries but only one returned --> prone to errors
            if len(multi_res) > 1:
                # warn user
                loggers.infstrct.warning(
                    (
                        'CAUTION: There are multiple subsystems which share the '
                        'same value >>%s<< for the '
                        'lookup property >>%s<<. '
                        'Only the first entry is returned.'
                    ),
                    lookup_val,
                    lookup_property,
                )

            return multi_res.iat[0]

    def lookup_custom_ID(
        self,
        system_type: SimSystemTypes,
        system_ID: SystemID,
    ) -> CustomID:
        id_prop: str
        match system_type:
            case SimSystemTypes.PRODUCTION_AREA:
                id_prop = 'prod_area_id'
            case SimSystemTypes.STATION_GROUP:
                id_prop = 'station_group_id'
            case SimSystemTypes.RESOURCE:
                id_prop = 'res_id'

        custom_id = cast(
            CustomID,
            self.lookup_subsystem_info(
                system_type=system_type,
                lookup_val=system_ID,
                lookup_property=id_prop,
                target_property='custom_id',
            ),
        )

        return custom_id

    def lookup_system_ID(
        self,
        system_type: SimSystemTypes,
        custom_ID: CustomID,
    ) -> SystemID:
        system = cast(
            System,
            self.lookup_subsystem_info(
                system_type=system_type,
                lookup_val=custom_ID,
                lookup_property='custom_id',
            ),
        )

        return system.system_id

    ####################################################################

    # [RESOURCES]
    @property
    def res_db(self) -> DataFrame:
        """obtain a current overview of registered objects in the environment"""
        return self._res_db

    @property
    def sinks(self) -> list[Sink]:
        """registered sinks"""
        return self._sinks

    @property
    def sink_registered(self) -> bool:
        return self._sink_registered

    def update_res_state(
        self,
        obj: InfrastructureObject,
        state: str,
        reset_temp: bool = False,
    ) -> None:
        """method to update the state of a resource object in the resource database"""
        loggers.infstrct.debug('Set state of >>%s<< to >>%s<<', obj, state)

        # check if 'TEMP' state should be reset
        if reset_temp:
            # special reset method, calls state setting to previous state
            obj.stat_monitor.reset_temp_state()
            state = obj.stat_monitor.state_current
        else:
            obj.stat_monitor.set_state(state=state)

        self._res_db.at[obj.system_id, 'state'] = state
        loggers.infstrct.debug('Executed state setting of >>%s<< to >>%s<<', obj, state)

    def res_objs_temp_state(
        self,
        res_objs: Iterable[InfrastructureObject],
        reset_temp: bool,
    ) -> None:
        """Sets/resets given resource objects from the 'TEMP' state

        Parameters
        ----------
        res_objs : Iterable[InfrastructureObject]
            objects for which the TEMP state should be changed
        set_temp : bool
            indicates if the temp state should be set or reset
        """
        for obj in res_objs:
            self.update_res_state(obj=obj, state='TEMP', reset_temp=reset_temp)
            # calculate KPIs if 'TEMP' state is set
            if not reset_temp:
                obj.stat_monitor.calc_KPI()

    def initialise(self) -> None:
        for prod_area in self._prod_area_db['prod_area']:
            prod_area.initialise()
        for station_group in self._station_group_db['station_group']:
            station_group.initialise()

    def finalise(self) -> None:
        # set end state for each resource object to calculate the right time amounts
        for res_obj in self._res_db['resource']:
            res_obj.finalise()
        loggers.infstrct.info(
            'Successful finalisation of the state information for all resource objects.'
        )

    def dashboard_update(self) -> None:
        # !! Placeholder, not implemented yet
        ...


class Dispatcher:
    def __init__(
        self,
        env: SimulationEnvironment,
        # sequencing_rule: str = 'FIFO',
        # allocation_rule: str = 'RANDOM',
    ) -> None:
        """
        Dispatcher class for given environment (only one dispatcher for each environment)
        - different functions to monitor all jobs in the environment
        - jobs report back their states to the dispatcher
        """

        # job data base as simple Pandas DataFrame
        # column data types
        self._job_prop: dict[str, type] = {
            'job_id': int,
            'custom_id': object,
            'job': object,
            'job_type': str,
            'prio': object,
            'total_proc_time': object,
            'creation_date': object,
            'release_date': object,
            'planned_starting_date': object,
            'actual_starting_date': object,
            'starting_date_deviation': object,
            'planned_ending_date': object,
            'actual_ending_date': object,
            'ending_date_deviation': object,
            'lead_time': object,
            'state': str,
        }
        self._job_db: DataFrame = pd.DataFrame(columns=list(self._job_prop.keys()))
        self._job_db: DataFrame = self._job_db.astype(self._job_prop)
        self._job_db: DataFrame = self._job_db.set_index('job_id')
        # properties by which a object can be obtained from the job database
        self._job_lookup_props: set[str] = set(['job_id', 'custom_id', 'name'])
        # properties which can be updated after creation
        self._job_update_props: set[str] = set(
            [
                'prio',
                'creation_date',
                'release_date',
                'planned_starting_date',
                'actual_starting_date',
                'starting_date_deviation',
                'planned_ending_date',
                'actual_ending_date',
                'ending_date_deviation',
                'lead_time',
                'state',
            ]
        )
        # date adjusted database for finalisation at the end of a simulation run
        self._job_db_date_adjusted = self._job_db.copy()

        # operation data base as simple Pandas DataFrame
        # column data types
        self._op_prop: dict[str, type] = {
            'op_id': int,
            'job_id': int,
            'custom_id': object,
            'op': object,
            'prio': object,
            'execution_system': object,
            'execution_system_custom_id': object,
            'execution_system_name': str,
            'execution_system_type': str,
            'target_station_custom_id': object,
            'target_station_name': str,
            'proc_time': object,
            'setup_time': object,
            'order_time': object,
            'creation_date': object,
            'release_date': object,
            'planned_starting_date': object,
            'actual_starting_date': object,
            'starting_date_deviation': object,
            'planned_ending_date': object,
            'actual_ending_date': object,
            'ending_date_deviation': object,
            'lead_time': object,
            'state': str,
        }
        self._op_db: DataFrame = pd.DataFrame(columns=list(self._op_prop.keys()))
        self._op_db: DataFrame = self._op_db.astype(self._op_prop)
        self._op_db: DataFrame = self._op_db.set_index('op_id')
        # properties by which a object can be obtained from the operation database
        self._op_lookup_props: set[str] = set(
            ['op_id', 'job_id', 'custom_id', 'name', 'machine']
        )
        # properties which can be updated after creation
        self._op_update_props: set[str] = set(
            [
                'prio',
                'target_station_custom_id',
                'target_station_name',
                'creation_date',
                'release_date',
                'actual_starting_date',
                'starting_date_deviation',
                'actual_ending_date',
                'ending_date_deviation',
                'lead_time',
                'state',
            ]
        )
        # date adjusted database for finalisation at the end of a simulation run
        self._op_db_date_adjusted = self._op_db.copy()

        # register in environment and get EnvID
        self._env = env
        # self._env.register_dispatcher(self)

        ####################################
        # managing IDs
        self._id_types = set(['job', 'op'])
        self._job_id_counter = LoadID(0)
        self._op_id_counter = LoadID(0)

        ####################################
        # priority rules
        self._sequencing_rules: frozenset[str] = frozenset(POLICIES_SEQ.keys())
        # set current priority rule
        # if sequencing_rule not in self._seq_rules:
        #     raise ValueError(
        #         (
        #             f'Sequencing rule {sequencing_rule} unknown. '
        #             f'Must be one of {self._seq_rules}'
        #         )
        #     )
        # else:
        #     self._curr_seq_rule = sequencing_rule

        # allocation rule
        self._allocation_rules: frozenset[str] = frozenset(POLICIES_ALLOC.keys())
        # # set current allocation rule
        # if allocation_rule not in self._allocation_rules:
        #     raise ValueError(
        #         (
        #             f'Allocation rule {allocation_rule} unknown. '
        #             f'Must be one of {self._allocation_rules}'
        #         )
        #     )
        # else:
        #     self._curr_alloc_rule = allocation_rule

        self._seq_rule: str | None = None
        self._alloc_rule: str | None = None
        self.seq_policy: GeneralPolicy | SequencingPolicy | None = None
        # = POLICIES_SEQ[self._curr_seq_rule]()
        self.alloc_policy: GeneralPolicy | AllocationPolicy | None = None
        # POLICIES_ALLOC[self._curr_alloc_rule]()

        # [STATS] cycle time
        self._cycle_time: Timedelta = Timedelta()

    ### DATA MANAGEMENT
    def __repr__(self) -> str:
        return f'Dispatcher(env: {self.env.name()})'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def seq_rule(self) -> str | None:
        return self._seq_rule

    @seq_rule.setter
    def seq_rule(
        self,
        rule: str,
    ) -> None:
        if rule not in self.sequencing_rules:
            raise ValueError(
                f'Priority rule {rule} unknown. Must be one of {self.sequencing_rules}'
            )
        else:
            self._seq_rule = rule
            self.seq_policy = POLICIES_SEQ[rule]
            loggers.dispatcher.info('Changed priority rule to %s', rule)

    @property
    def sequencing_rules(self) -> frozenset[str]:
        return self._sequencing_rules

    @property
    def allocation_rules(self) -> frozenset[str]:
        return self._allocation_rules

    @property
    def alloc_rule(self) -> str | None:
        return self._alloc_rule

    @alloc_rule.setter
    def alloc_rule(
        self,
        rule: str,
    ) -> None:
        if rule not in self.allocation_rules:
            raise ValueError(
                f'Allocation rule {rule} unknown. Must be one of {self.allocation_rules}'
            )
        else:
            self._alloc_rule = rule
            self.alloc_policy = POLICIES_ALLOC[rule]
            loggers.dispatcher.info('Changed allocation rule to >>%s<<', rule)

    def _obtain_load_obj_id(
        self,
        load_type: str,
    ) -> LoadID:
        """Simple counter function for managing operation IDs"""
        # assign id and set counter up
        if load_type not in self._id_types:
            raise ValueError(f"Given type {type} not valid. Choose from '{self._id_types}'")

        load_id: LoadID
        match load_type:
            case 'job':
                load_id = self._job_id_counter
                self._job_id_counter += 1
            case 'op':
                load_id = self._op_id_counter
                self._op_id_counter += 1

        return load_id

    @property
    def cycle_time(self) -> Timedelta:
        return self._cycle_time

    def _calc_cycle_time(self) -> None:
        """
        Obtaining the current cycle time of all operations
        """
        self._cycle_time: Timedelta = (
            self._op_db['actual_ending_date'].max() - self._env.starting_datetime
        )

    ### JOBS ###
    def register_job(
        self,
        job: Job,
        custom_identifier: CustomID | None,
        state: str,
    ) -> tuple[SimulationEnvironment, LoadID]:
        """
        registers an job object in the dispatcher instance by assigning an unique id and
        adding the object to the associated jobs
        """
        # obtain id
        job_id = self._obtain_load_obj_id(load_type='job')

        # time of creation
        # creation_date = self.env.now()
        creation_date = self.env.t_as_dt()

        # new entry for job data base
        new_entry: DataFrame = pd.DataFrame(
            {
                'job_id': [job_id],
                'custom_id': [custom_identifier],
                'job': [job],
                'job_type': [job.job_type],
                'prio': [job.prio],
                'total_proc_time': [job.total_proc_time],
                'creation_date': [creation_date],
                'release_date': [job.time_release],
                'planned_starting_date': [job.time_planned_starting],
                'actual_starting_date': [job.time_actual_starting],
                'starting_date_deviation': [job.starting_date_deviation],
                'planned_ending_date': [job.time_planned_ending],
                'actual_ending_date': [job.time_actual_ending],
                'ending_date_deviation': [job.ending_date_deviation],
                'lead_time': [job.lead_time],
                'state': [state],
            }
        )
        new_entry = new_entry.astype(self._job_prop)
        new_entry = new_entry.set_index('job_id')
        self._job_db = pd.concat([self._job_db, new_entry])

        loggers.dispatcher.info('Successfully registered job with JobID >>%s<<', job_id)

        # write job information directly
        job.time_creation = creation_date

        # return current env, job ID, job name
        return self._env, job_id

    def update_job_db(
        self,
        job: Job,
        property: str,
        val: Any,
    ) -> None:
        """
        updates the information of a job for a given property
        """
        # check if property is a filter criterion
        if property not in self._job_update_props:
            raise IndexError(
                f"Property '{property}' is not allowed. Choose from {self._job_update_props}"
            )
        # None type value can not be set
        if val is None:
            raise TypeError('The set value can not be of type >>None<<.')

        self._job_db.at[job.job_id, property] = val

    def release_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the release of the given job
        necessary for time statistics
        """
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        job.time_release = current_time
        job.is_released = True
        self.update_job_db(job=job, property='release_date', val=job.time_release)

    def enter_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the start of the given job on the first Processing Station
        necessary for time statistics
        """
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # starting time processing
        job.time_actual_starting = current_time

        # starting times
        if job.time_planned_starting is not None:
            job.starting_date_deviation = job.time_actual_starting - job.time_planned_starting
            self.update_job_db(
                job=job, property='starting_date_deviation', val=job.starting_date_deviation
            )

        # update operation database
        self.update_job_db(
            job=job, property='actual_starting_date', val=job.time_actual_starting
        )

    def finish_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the exit of the given job
        necessary for time statistics
        """
        # [STATS]
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # job.time_exit = current_time
        job.time_actual_ending = current_time
        job.is_finished = True
        # job.lead_time = job.time_exit - job.time_release
        job.lead_time = job.time_actual_ending - job.time_release

        # ending times
        if job.time_planned_ending is not None:
            job.ending_date_deviation = job.time_actual_ending - job.time_planned_ending
            self.update_job_db(
                job=job, property='ending_date_deviation', val=job.ending_date_deviation
            )

        # update databases
        self.update_job_state(job=job, state='FINISH')
        # self.update_job_db(job=job, property='exit_date', val=job.time_exit)
        self.update_job_db(job=job, property='actual_ending_date', val=job.time_actual_ending)
        self.update_job_db(job=job, property='lead_time', val=job.lead_time)
        # [MONITOR] finalise stats
        job.stat_monitor.finalise_stats()

    def update_job_process_info(
        self,
        job: Job,
        preprocess: bool,
    ) -> None:
        """
        method to write necessary information of the job and its current operation
        before and after processing,
        invoked by Infrastructure Objects
        """
        # get current operation of the job instance
        current_op = job.current_op
        # before processing
        if current_op is None:
            raise ValueError(
                'Can not update job info because current operation is not available.'
            )
        elif preprocess:
            # operation enters Processing Station
            # if first operation of given job add job's starting information
            if job.num_finished_ops == 0:
                self.enter_job(job=job)
            self.enter_operation(op=current_op)
        else:
            # after processing
            self.finish_operation(op=current_op)
            job.num_finished_ops += 1

    def update_job_state(
        self,
        job: Job,
        state: str,
    ) -> None:
        """method to update the state of a job in the job database"""
        # update state tracking of the job instance
        job.stat_monitor.set_state(state=state)
        # update job database
        self.update_job_db(job=job, property='state', val=state)
        # only update operation state if it is not finished
        # operations are finished by post-process call to their 'finalise' method

        # update state of the corresponding operation
        if job.current_op is not None:
            self.update_operation_state(op=job.current_op, state=state)

    def get_next_operation(
        self,
        job: Job,
    ) -> Operation | None:
        """
        get next operation of given job
        """
        # last operation information
        job.last_op = job.current_op
        job.last_proc_time = job.current_proc_time
        job.last_setup_time = job.current_setup_time
        job.last_order_time = job.current_order_time
        # current operation information
        if job.open_operations:
            op = job.open_operations.popleft()
            job.current_proc_time = op.proc_time
            job.current_setup_time = op.setup_time
            job.current_order_time = op.order_time
            # only reset job prio if there are OP-wise defined priorities
            if job.op_wise_prio:
                if op.prio is None:
                    raise ValueError(f'Operation {op} has no priority defined.')
                job.prio = op.prio  # use setter function to catch possible errors
                self.update_job_db(job=job, property='prio', val=job.prio)
            if job.op_wise_starting_date:
                job.time_planned_starting = op.time_planned_starting
                self.update_job_db(
                    job=job, property='planned_starting_date', val=job.time_planned_starting
                )
            if job.op_wise_ending_date:
                job.time_planned_ending = op.time_planned_ending
                self.update_job_db(
                    job=job, property='planned_ending_date', val=job.time_planned_ending
                )
        else:
            op = None
            job.current_proc_time = None
            job.current_setup_time = None
            job.current_order_time = None

        job.current_op = op

        return op

    ### OPERATIONS ###
    def register_operation(
        self,
        op: Operation,
        exec_system_identifier: SystemID,
        target_station_group_identifier: SystemID | None,
        custom_identifier: CustomID | None,
        state: str,
    ) -> LoadID:
        """
        registers an operation object in the dispatcher instance by assigning an unique id and
        adding the object to the associated operations

        obj: operation to register
        machine_identifier: custom ID of the associated machine (user interface)
        custom_identifier: custom identifier of the operation
            (kept for consistency reasons, perhaps remove later)
        name: assigned name the operation
        status: for future features if status of operations is tracked

        outputs:
        op_id: assigned operation ID
        name: assigned name
        machine: corresponding machine infrastructure object
        test
        """
        # infrastructure manager
        infstruct_mgr = self.env.infstruct_mgr
        # obtain id
        op_id = self._obtain_load_obj_id(load_type='op')
        # time of creation
        # creation_date = self.env.now()
        creation_date = self.env.t_as_dt()

        # setup time
        setup_time: Timedelta
        if op.setup_time is not None:
            setup_time = op.setup_time
        else:
            setup_time = Timedelta()

        # corresponding execution system in which the operation is performed
        # no pre-determined assignment of processing stations
        exec_system = cast(
            ProductionArea,
            infstruct_mgr.lookup_subsystem_info(
                system_type=EXEC_SYSTEM_TYPE, lookup_val=exec_system_identifier
            ),
        )
        # if target station group is specified, get instance
        target_station_group: StationGroup | None
        if target_station_group_identifier is not None:
            target_station_group = cast(
                StationGroup,
                infstruct_mgr.lookup_subsystem_info(
                    system_type=SimSystemTypes.STATION_GROUP,
                    lookup_val=target_station_group_identifier,
                ),
            )
            # validity check: only target stations allowed which are
            # part of the current execution system
            if target_station_group.system_id not in exec_system.subsystems:
                raise ValueError(f'{target_station_group} is not part of {exec_system}. \
                    Mismatch between execution system and associated station groups.')
        else:
            target_station_group = None

        # new entry for operation data base
        new_entry: DataFrame = pd.DataFrame(
            {
                'op_id': [op_id],
                'job_id': [op.job_id],
                'custom_id': [custom_identifier],
                'op': [op],
                'prio': [op.prio],
                'execution_system': [exec_system],
                'execution_system_custom_id': [exec_system.custom_identifier],
                'execution_system_name': [exec_system.name],
                'execution_system_type': [exec_system.system_type],
                'target_station_custom_id': [None],
                'target_station_name': [None],
                'proc_time': [op.proc_time],
                'setup_time': [setup_time],
                'order_time': [op.order_time],
                'creation_date': [creation_date],
                'release_date': [op.time_release],
                'planned_starting_date': [op.time_planned_starting],
                'actual_starting_date': [op.time_actual_starting],
                'starting_date_deviation': [op.starting_date_deviation],
                'planned_ending_date': [op.time_planned_ending],
                'actual_ending_date': [op.time_actual_ending],
                'ending_date_deviation': [op.ending_date_deviation],
                'lead_time': [op.lead_time],
                'state': [state],
            }
        )
        new_entry: DataFrame = new_entry.astype(self._op_prop)
        new_entry = new_entry.set_index('op_id')
        self._op_db = pd.concat([self._op_db, new_entry])

        loggers.dispatcher.info('Successfully registered operation with OpID >>%s<<', op_id)

        # write operation information directly
        op.target_exec_system = exec_system
        op.target_station_group = target_station_group
        op.time_creation = creation_date

        # return operation ID
        return op_id

    def update_operation_db(
        self,
        op: Operation,
        property: str,
        val: Any,
    ) -> None:
        """
        updates the information of a job for a given property
        """
        # check if property is a filter criterion
        if property not in self._op_update_props:
            raise IndexError(
                f"Property '{property}' is not allowed. Choose from {self._op_update_props}"
            )
        # None type value can not be looked for
        if val is None:
            raise TypeError("The lookup value can not be of type 'None'.")

        self._op_db.at[op.op_id, property] = val

    def update_operation_state(
        self,
        op: Operation,
        state: str,
    ) -> None:
        """method to update the state of a operation in the operation database"""
        # update state tracking of the operation instance
        op.stat_monitor.set_state(state=state)
        # update operation database
        self.update_operation_db(op=op, property='state', val=state)

    def release_operation(
        self,
        op: Operation,
        target_station: ProcessingStation,
    ) -> None:
        """
        used to signal the release of the given operation
        necessary for time statistics
        """
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # release time
        op.time_release = current_time
        op.is_released = True
        # update operation database
        # release date
        self.update_operation_db(op=op, property='release_date', val=op.time_release)
        # target station: custom identifier + name
        self.update_operation_db(
            op=op, property='target_station_custom_id', val=target_station.custom_identifier
        )
        self.update_operation_db(
            op=op, property='target_station_name', val=target_station.name
        )

    def enter_operation(
        self,
        op: Operation,
    ) -> None:
        """
        used to signal the start of the given operation on a Processing Station
        necessary for time statistics
        """
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # starting time processing
        # op.time_entry = current_time
        op.time_actual_starting = current_time

        # starting times
        if op.time_planned_starting is not None:
            op.starting_date_deviation = op.time_actual_starting - op.time_planned_starting
            self.update_operation_db(
                op=op, property='starting_date_deviation', val=op.starting_date_deviation
            )

        # update operation database
        # self.update_operation_db(op=op, property='entry_date', val=op.time_entry)
        self.update_operation_db(
            op=op, property='actual_starting_date', val=op.time_actual_starting
        )

    def finish_operation(
        self,
        op: Operation,
    ) -> None:
        """
        used to signal the finalisation of the given operation
        necessary for time statistics
        """
        # current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # [STATE] finished
        op.is_finished = True
        # [STATS] end + lead time
        # op.time_exit = current_time
        op.time_actual_ending = current_time
        # op.lead_time = op.time_exit - op.time_release
        op.lead_time = op.time_actual_ending - op.time_release

        # ending times
        if op.time_planned_ending is not None:
            op.ending_date_deviation = op.time_actual_ending - op.time_planned_ending
            self.update_operation_db(
                op=op, property='ending_date_deviation', val=op.ending_date_deviation
            )

        # update databases
        loggers.dispatcher.debug(
            'Update databases for OP %s ID %s with [%s, %s]',
            op,
            op.op_id,
            op.time_actual_ending,
            op.lead_time,
        )
        self.update_operation_state(op=op, state='FINISH')
        # self.update_operation_db(op=op, property='exit_date', val=op.time_exit)
        self.update_operation_db(
            op=op, property='actual_ending_date', val=op.time_actual_ending
        )
        self.update_operation_db(op=op, property='lead_time', val=op.lead_time)

        # [MONITOR] finalise stats
        op.stat_monitor.finalise_stats()

    ### PROPERTIES ###
    @property
    def job_db(self) -> DataFrame:
        """
        obtain a current overview of registered jobs in the environment
        """
        return self._job_db

    @property
    def job_db_date_adjusted(self) -> DataFrame:
        """
        obtain a current date adjusted overview of registered jobs in
        the environment
        """
        return self._job_db_date_adjusted

    @property
    def op_db(self) -> DataFrame:
        """
        obtain a current overview of registered operations in the environment
        """
        return self._op_db

    @property
    def op_db_date_adjusted(self) -> DataFrame:
        """
        obtain a current date adjusted overview of registered
        operations in the environment
        """
        return self._op_db_date_adjusted

    # @lru_cache(maxsize=200)
    def lookup_job_obj_prop(
        self,
        val: LoadID | CustomID | str,
        property: str = 'job_id',
        target_prop: str = 'job',
    ) -> Any:
        """
        obtain a job object from the dispatcher by its property and corresponding value
        properties: job_id, custom_id, name
        """
        # check if property is a filter criterion
        if property not in self._job_lookup_props:
            raise IndexError(
                f"Property '{property}' is not allowed. Choose from {self._job_lookup_props}"
            )
        # None type value can not be looked for
        if val is None:
            raise TypeError("The lookup value can not be of type 'None'.")

        # filter resource database for prop-value pair
        if property == 'job_id':
            # direct indexing for ID property; job_id always unique,
            # no need for duplicate check
            try:
                idx_res: Any = self._job_db.at[val, target_prop]
                return idx_res
            except KeyError:
                raise IndexError(
                    (
                        f'There were no jobs found for the '
                        f'property >>{property}<< '
                        f'with the value >>{val}<<'
                    )
                )
        else:
            multi_res = self._job_db.loc[self._job_db[property] == val, target_prop]
            # check for empty search result, at least one result necessary
            if len(multi_res) == 0:
                raise IndexError(
                    (
                        f'There were no jobs found for the property >>{property}<< '
                        f'with the value >>{val}<<'
                    )
                )
            # check for multiple entries with same prop-value pair
            ########### PERHAPS CHANGE NECESSARY
            ### multiple entries but only one returned --> prone to errors
            elif len(multi_res) > 1:
                # warn user
                loggers.dispatcher.warning(
                    (
                        'CAUTION: There are multiple jobs which share the '
                        'same value >>%s<< for the property >>%s<<. '
                        'Only the first entry is returned.'
                    ),
                    val,
                    property,
                )

            return multi_res.iat[0]

    ### ROUTING LOGIC ###

    def check_alloc_dispatch(
        self,
        job: Job,
    ) -> tuple[bool, AllocationAgent | None]:
        # get next operation of job
        next_op = self.get_next_operation(job=job)
        is_agent: bool = False
        agent: AllocationAgent | None = None
        if self.alloc_rule == 'AGENT' and next_op is not None:
            if next_op.target_exec_system is None:
                # should never happen as each operation is registered with
                # a system instance
                raise ValueError('No target execution system assigned.')
            else:
                # check agent availability
                is_agent = next_op.target_exec_system.check_alloc_agent()
                if is_agent:
                    agent = next_op.target_exec_system.alloc_agent
                    agent.action_feasible = False
                else:
                    raise ValueError(
                        'Allocation rule set to agent, but no agent instance found.'
                    )

        return is_agent, agent

    def request_agent_alloc(
        self,
        job: Job,
    ) -> None:
        # get the target operation of the job
        op = job.current_op
        if op is None:
            raise ValueError(f'Current operation of job {job} not available.')
        # execution system of current OP
        target_exec_system = op.target_exec_system
        if target_exec_system is None:
            raise ValueError(f'No target execution system assigned for Operation {op}.')
        # agent available, get necessary information for decision
        agent = target_exec_system.alloc_agent

        # [KPI] calculate necessary KPIs by putting associated
        # objects in TEMP state
        self.env.infstruct_mgr.res_objs_temp_state(
            res_objs=agent.assoc_proc_stations,
            reset_temp=False,
        )

        # request decision from agent, sets internal flag
        agent.request_decision(job=job, op=op)
        loggers.dispatcher.debug('[DISPATCHER] Alloc Agent: Decision request made.')

        # reset TEMP state
        self.env.infstruct_mgr.res_objs_temp_state(
            res_objs=agent.assoc_proc_stations,
            reset_temp=True,
        )

    def request_job_allocation(
        self,
        job: Job,
        is_agent: bool,
    ) -> InfrastructureObject:
        """
        request an allocation decision for the given job
        (determine the next processing station on which the job shall be placed)

        1. obtaining the target station group
        2. select from target station group
        3. return target station (InfrastructureObject)

        requester: output side infrastructure object
        request for: infrastructure object instance
        """

        loggers.dispatcher.debug('[DISPATCHER] REQUEST TO DISPATCHER FOR ALLOCATION')

        ## NEW TOP-DOWN-APPROACH
        # routing of jobs is now organized in a hierarchical fashion and can be described
        # for each hierarchy level separately
        # routing in Production Areas --> Station Groups --> Processing Stations
        # so each job must contain information about the production areas and
        # the corresponding station groups

        ## choice from station group stays as method
        # routing essentially depending on production areas --> JOB FROM AREA TO AREA
        # NOW DIFFERENTIATE:
        ### ~~(1) choice between station groups of the current area~~
        #           placement on machines outside the station group not possible
        ### (2) choice between processing stations of the current area
        #           placement on machines outside the station group possible,
        #           but the stations could be filtered by their station group IDs
        ### --> (2) as implemented solution

        # get the target operation of the job
        # next_op = self.get_next_operation(job=job)
        op = job.current_op
        if op is not None:
            # get target execution system ((sub)system type)
            # (defined by the global variable EXEC_SYSTEM_TYPE)
            target_exec_system = op.target_exec_system
            if target_exec_system is None:
                raise ValueError('No target execution system assigned.')
            # get target station group
            target_station_group = op.target_station_group
            if target_station_group is None:
                raise ValueError('No target station group assigned.')

            loggers.dispatcher.debug('[DISPATCHER] Next operation %s', op)
            # obtain target station (InfrastructureObject)
            target_station = self._choose_target_station_from_exec_system(
                exec_system=target_exec_system,
                op=op,
                is_agent=is_agent,
                target_station_group=target_station_group,
            )

            # with allocation request operation is released
            self.release_operation(op=op, target_station=target_station)
        # all operations done, look for sinks
        else:
            infstruct_mgr = self.env.infstruct_mgr
            sinks = infstruct_mgr.sinks
            # ?? [PERHAPS CHANGE IN FUTURE]
            # use first sink of the registered ones
            target_station = sinks[0]

        loggers.dispatcher.debug(
            '[DISPATCHER] Next operation is %s with machine group (machine) %s',
            op,
            target_station,
        )

        return target_station

    def _choose_target_station_from_exec_system(
        self,
        exec_system: System,
        op: Operation,
        is_agent: bool,
        target_station_group: StationGroup | None = None,
    ) -> ProcessingStation:
        infstruct_mgr = self.env.infstruct_mgr

        if not is_agent:
            if target_station_group:
                # preselection of station group only with allocation rules
                # other than >>AGENT<<
                # returned ProcessingStations automatically feasible
                # regarding their StationGroup
                stations = target_station_group.assoc_proc_stations
            else:
                stations = exec_system.assoc_proc_stations

            # [KPIs] calculate necessary information for decision making
            # put all associated processing stations of that group in 'TEMP' state
            infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=False)
            candidates = tuple(ps for ps in stations if ps.stat_monitor.is_available)
            # if there are no available ones: use all stations
            if candidates:
                avail_stations = candidates
            else:
                avail_stations = stations

            # ** Allocation Rules
            # first use StationGroup, then ExecutionSystem, then Dispatcher (global)
            policy: GeneralPolicy | AllocationPolicy
            if (
                target_station_group is not None
                and target_station_group.alloc_policy is not None
            ):
                policy = target_station_group.alloc_policy
            elif exec_system.alloc_policy is not None:
                policy = exec_system.alloc_policy
            elif self.alloc_policy is not None:
                policy = self.alloc_policy
            else:
                raise ValueError('No allocation policy defined.')

            target_station = policy.apply(items=avail_stations)

            # TODO add policy-based rule sets
            # apply different strategies to select a station out of the station group
            # match self._curr_alloc_rule:
            #     case 'RANDOM':
            #         # [RANDOM CHOICE]
            #         target_station: ProcessingStation = random.choice(avail_stations)
            #     case 'UTILISATION':
            #         # [UTILISATION]
            #         # choose the station with the lowest utilisation to time
            #         target_station: ProcessingStation = min(
            #             avail_stations, key=attrgetter('stat_monitor.utilisation')
            #         )
            #         loggers.dispatcher.debug(
            #             (
            #                 f'[DISPATCHER: {self}] Utilisation of '
            #                 f'{target_station=} is '
            #                 f'{target_station.stat_monitor.utilisation:.4f}'
            #             )
            #         )
            #     case 'WIP_LOAD_TIME':
            #         # WIP as load/processing time, choose station with lowest WIP
            #         target_station: ProcessingStation = min(
            #             avail_stations, key=attrgetter('stat_monitor.WIP_load_time')
            #         )
            #         loggers.dispatcher.debug(
            #             (
            #                 f'[DISPATCHER: {self}] WIP LOAD TIME of '
            #                 f'{target_station=} is '
            #                 f'{target_station.stat_monitor.WIP_load_time}'
            #             )
            #         )
            #     case 'WIP_LOAD_JOBS':
            #         # WIP as number of associated jobs, choose station with lowest WIP
            #         target_station: ProcessingStation = min(
            #             avail_stations, key=attrgetter('stat_monitor.WIP_load_num_jobs')
            #         )
            #         loggers.dispatcher.debug(
            #             (
            #                 f'[DISPATCHER: {self}] WIP LOAD NUM JOBS of '
            #                 f'{target_station=} is '
            #                 f'{target_station.stat_monitor.WIP_load_time:.2f}'
            #             )
            #         )
            # [KPIs] reset all associated processing stations of that group
            # to their original state
            infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=True)
        else:
            # ** AGENT decision
            # available stations
            # agent can choose from all associated stations, not only available ones
            # availability of processing stations should be learned by the agent
            agent = exec_system.alloc_agent
            avail_stations = agent.assoc_proc_stations
            # Feature vector already built when request done to agent
            # get chosen station by tuple index (agent's action)
            station_idx = agent.action
            if station_idx is None:
                raise ValueError('No station index chosen')
            target_station = avail_stations[station_idx]
            agent.action_feasible = self._env.check_feasible_agent_alloc(
                target_station=target_station, op=op
            )
            loggers.agents.debug('Action feasibility status: %s', agent.action_feasible)

        return target_station

    # TODO change return type to only job
    def request_job_sequencing(
        self,
        req_obj: InfrastructureObject,
    ) -> tuple[Job, Timedelta, Timedelta | None]:
        """
        request a sequencing decision for a given queue of the requesting resource
        requester: input side processing stations
        request for: job instance

        req_obj: requesting object (ProcessingStation)
        """
        # SIGNALING SEQUENCING DECISION
        # (ONLY IF MULTIPLE JOBS IN THE QUEUE EXIST)
        ## theoretically: get logic queue of requesting object -->
        # information about feasible jobs -->
        ## [*] choice of sequencing agent (based on which properties?)
        # --> preparing feature vector as input -->
        ## trigger agent decision --> map decision to feasible jobs
        ## [*] use implemented priority rules as intermediate step

        loggers.dispatcher.info('[DISPATCHER] REQUEST TO DISPATCHER FOR SEQUENCING')

        # get logic queue of requesting object
        # contains all feasible jobs for this resource
        logic_queue = req_obj.logic_queue
        # get job from logic queue with currently defined priority rule
        job = self._seq_priority_rule(req_obj=req_obj, queue=logic_queue)
        # reset environment signal for SEQUENCING
        if job.current_proc_time is None:
            raise ValueError(f'No processing time defined for job {job}.')

        return job, job.current_proc_time, job.current_setup_time

    # TODO policy-based decision making
    def _seq_priority_rule(
        self,
        req_obj: InfrastructureObject,
        queue: salabim.Queue,
    ) -> Job:
        """apply priority rules to a pool of jobs"""

        # ** Allocation Rules
        # first use requesting object, then Dispatcher (global)
        policy: GeneralPolicy | SequencingPolicy
        if req_obj.seq_policy is not None:
            policy = req_obj.seq_policy
        elif self.seq_policy is not None:
            policy = self.seq_policy
        else:
            raise ValueError('No sequencing policy defined.')

        job_collection = cast(list[Job], queue.as_list())
        job = policy.apply(items=job_collection)
        queue.remove(job)

        # match self._curr_prio_rule:
        #     # first in, first out
        #     case 'FIFO':
        #         # salabim queue pops first entry if no index is specified,
        #         # not last like in Python
        #         job = cast(Job, queue.pop())
        #     # last in, last out
        #     case 'LIFO':
        #         # salabim queue pops first entry if no index is specified,
        #         # not last like in Python
        #         job = cast(Job, queue.pop(-1))
        #     # shortest processing time
        #     case 'SPT':
        #         # choose job with shortest processing time
        #         temp = cast(list[Job], queue.as_list())
        #         job = min(temp, key=attrgetter('current_proc_time'))
        #         # remove job from original queue
        #         queue.remove(job)
        #     # longest processing time
        #     case 'LPT':
        #         # choose job with longest processing time
        #         temp = cast(list[Job], queue.as_list())
        #         job = max(temp, key=attrgetter('current_proc_time'))
        #         # remove job from original queue
        #         queue.remove(job)
        #     # shortest setup time
        #     case 'SST':
        #         # choose job with shortest setup time
        #         temp = cast(list[Job], queue.as_list())
        #         job = min(temp, key=attrgetter('current_setup_time'))
        #         # remove job from original queue
        #         queue.remove(job)
        #     # longest setup time
        #     case 'LST':
        #         # choose job with longest setup time
        #         temp = cast(list[Job], queue.as_list())
        #         job = max(temp, key=attrgetter('current_setup_time'))
        #         # remove job from original queue
        #         queue.remove(job)
        #     case 'PRIO':
        #         # choose job with highest priority
        #         temp = cast(list[Job], queue.as_list())
        #         job = max(temp, key=attrgetter('prio'))
        #         # remove job from original queue
        #         queue.remove(job)

        return job

    ### ANALYSE ###
    def draw_gantt_chart(
        self,
        use_custom_proc_station_id: bool = True,
        sort_by_proc_station: bool = False,
        sort_ascending: bool = True,
        group_by_exec_system: bool = False,
        dates_to_local_tz: bool = True,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'gantt_chart',
    ) -> PlotlyFigure:
        """
        draw a Gantt chart based on the dispatcher's operation database
        use_custom_machine_id: whether to use the custom IDs of the
        processing station (True) or its name (False)
        sort_by_proc_station: whether to sort by processing station
        property (True) or by job name (False)
            default: False
        sort_ascending: whether to sort in ascending (True) or
            descending order (False)
            default: True
        use_duration: plot each operation with its scheduled duration instead of
            the delta time
            between start and end; if there were no interruptions both methods return
            the same results
            default: False
        """
        # TODO: Implement debug function to display results during sim runs

        # filter operation DB for relevant information
        filter_items: list[str] = [
            'job_id',
            'target_station_custom_id',
            'target_station_name',
            'execution_system',
            'execution_system_custom_id',
            'prio',
            'planned_starting_date',
            'actual_starting_date',
            'planned_ending_date',
            'actual_ending_date',
            'proc_time',
            'setup_time',
            'order_time',
        ]

        hover_data: dict[str, str | bool] = {
            'job_id': False,
            'target_station_custom_id': True,
            'execution_system_custom_id': True,
            'prio': True,
            'planned_starting_date': True,
            'actual_starting_date': True,
            'planned_ending_date': True,
            'actual_ending_date': True,
            'proc_time': True,
            'setup_time': True,
            'order_time': True,
        }
        # TODO: disable hover infos if some entries are None

        # hover_template: str = "proc_time: %{proc_time|%d:%H:%M:%S}"
        # TODO: use dedicated method to transform dates of job and op databases
        if dates_to_local_tz:
            self._job_db_date_adjusted = adjust_db_dates_local_tz(db=self._job_db)
            self._op_db_date_adjusted = adjust_db_dates_local_tz(db=self._op_db)
            target_db = self._op_db_date_adjusted
        else:
            target_db = self._op_db

        # filter only finished operations (for debug display)
        target_db = target_db.loc[(target_db['state'] == 'FINISH')]

        df = target_db.filter(items=filter_items)
        # calculate delta time between start and end
        # Timedelta
        df['delta'] = df['actual_ending_date'] - df['actual_starting_date']

        # choose relevant processing station property
        proc_station_prop: str
        if use_custom_proc_station_id:
            proc_station_prop = 'target_station_custom_id'
        else:
            proc_station_prop = 'target_station_name'

        # check if sorting by processing station is wanted and custom ID should be used or not
        # sorting
        sort_key: str
        if sort_by_proc_station:
            sort_key = proc_station_prop
        else:
            sort_key = 'job_id'

        df['job_id'] = df['job_id'].astype(str)
        df = df.sort_values(by=sort_key, ascending=sort_ascending, kind='stable')

        # group by value
        if group_by_exec_system:
            group_by_key = 'execution_system_custom_id'
        else:
            group_by_key = 'job_id'

        # build Gantt chart with Plotly Timeline
        fig: PlotlyFigure = px.timeline(
            df,
            x_start='actual_starting_date',
            x_end='actual_ending_date',
            y=proc_station_prop,
            color=group_by_key,
            hover_name='job_id',
            hover_data=hover_data,
        )
        fig.update_yaxes(type='category', autorange='reversed')

        if self._env.debug_dashboard:
            # send by websocket
            fig_json = cast(str | None, plotly.io.to_json(fig=fig))
            if fig_json is None:
                raise ValueError('Could not convert figure to JSON. Returned >>None<<.')
            self._env.ws_con.send(fig_json)
        else:
            fig.show()
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)

        if save_img:
            file = f'{file_name}'
            fig.write_image(file)

        return fig

    def initialise(self) -> None:
        # !! Placeholder, do nothing at the moment
        pass

    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by
        the environment's "finalise_sim" method
        """
        self._calc_cycle_time()
        self._job_db_date_adjusted = adjust_db_dates_local_tz(db=self._job_db)
        self._op_db_date_adjusted = adjust_db_dates_local_tz(db=self._op_db)

    def dashboard_update(self) -> None:
        """
        method to be called by the environment's "update_dashboard" method
        """
        # !! Placeholder, do nothing at the moment
        pass


# ** systems


class System:
    def __init__(
        self,
        env: SimulationEnvironment,
        system_type: SimSystemTypes,
        custom_identifier: CustomID,
        abstraction_level: int,
        name: str | None = None,
        state: str | None = None,
    ) -> None:
        # [BASIC INFO]
        # environment
        self._env = env
        self._system_type = system_type
        # subsystem information
        self.subsystems: dict[SystemID, System] = {}
        self.subsystems_ids: set[SystemID] = set()
        self.subsystems_custom_ids: set[CustomID] = set()
        # supersystem information
        self.supersystems: dict[SystemID, System] = {}
        self.supersystems_ids: set[SystemID] = set()
        self.supersystems_custom_ids: set[CustomID] = set()
        # number of lower levels, how many levels of subsystems are possible
        self._abstraction_level = abstraction_level
        # collection of all associated ProcessingStations
        self._assoc_proc_stations: tuple[ProcessingStation, ...] = ()
        self._num_assoc_proc_stations: int = 0
        # indicator if the system contains processing stations
        self._containing_proc_stations: bool = False

        infstruct_mgr = self.env.infstruct_mgr
        self._system_id, self._name = infstruct_mgr.register_subsystem(
            system_type=self._system_type,
            obj=self,
            custom_identifier=custom_identifier,
            name=name,
            state=state,
        )
        self._custom_identifier = custom_identifier

        self.seq_policy: GeneralPolicy | SequencingPolicy | None = None
        self.alloc_policy: GeneralPolicy | AllocationPolicy | None = None

        # [AGENT] decision agent
        self._agent_types: set[str] = set(['SEQ', 'ALLOC'])
        self._alloc_agent_registered: bool = False
        # assignment
        self._alloc_agent: AllocationAgent | None = None

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def assoc_proc_stations(self) -> tuple[ProcessingStation, ...]:
        return self._assoc_proc_stations

    @property
    def num_assoc_proc_station(self) -> int:
        return self._num_assoc_proc_stations

    ### REWORK
    def register_agent(
        self,
        agent: Agent,
        agent_task: AgentTasks,
    ) -> tuple[Self, SimulationEnvironment]:
        if agent_task not in self._agent_types:
            raise ValueError(
                (
                    f'The agent type >>{agent_task}<< is not allowed. '
                    f'Choose from {self._agent_types}'
                )
            )

        match agent_task:
            case 'ALLOC':
                # allocation agents on lowest hierarchy level not allowed
                if self._abstraction_level == 0:
                    raise RuntimeError(
                        (
                            'Can not register allocation agents '
                            'for lowest hierarchy level objects.'
                        )
                    )
                # registration, type and existence check
                if not self._alloc_agent_registered and isinstance(agent, AllocationAgent):
                    self._alloc_agent = agent
                    self._alloc_agent_registered = True
                    loggers.pyf_env.info(
                        'Successfully registered Allocation Agent in %s',
                        self,
                    )
                elif not isinstance(agent, AllocationAgent):
                    raise TypeError(
                        (
                            f'The object must be of type >>AllocationAgent<< '
                            f'but is type >>{type(agent)}<<'
                        )
                    )
                else:
                    raise AttributeError(
                        (
                            'There is already a registered AllocationAgent instance. '
                            'Only one instance per system is allowed.'
                        )
                    )
            case 'SEQ':
                raise NotImplementedError(
                    'Registration of sequencing agents not supported yet!'
                )

        return self, self.env

    @property
    def alloc_agent(self) -> AllocationAgent:
        if self._alloc_agent is None:
            raise ValueError('No AllocationAgent instance registered.')
        else:
            return self._alloc_agent

    def check_alloc_agent(self) -> bool:
        """checks if an allocation agent is registered for the system"""
        if self._alloc_agent_registered:
            return True
        else:
            return False

    def __str__(self) -> str:
        return (
            f'System (type: {self._system_type}, '
            f'custom_id: {self._custom_identifier}, name: {self._name})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple[SystemID, str]:
        return (self._system_id, self._system_type)

    def __hash__(self) -> int:
        return hash(self.__key())

    @property
    def system_type(self) -> str:
        return self._system_type

    @property
    def system_id(self) -> SystemID:
        return self._system_id

    @property
    def custom_identifier(self) -> CustomID:
        return self._custom_identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def abstraction_level(self) -> int:
        return self._abstraction_level

    @property
    def containing_proc_stations(self) -> bool:
        return self._containing_proc_stations

    @containing_proc_stations.setter
    def containing_proc_stations(
        self,
        val: bool,
    ) -> None:
        if not isinstance(val, bool):
            raise TypeError(f'Type of {val} must be boolean, but is {type(val)}')

        self._containing_proc_stations = val

    def subsystems_as_list(self) -> list[System]:
        """output the associated subsystems as list

        Returns
        -------
        list[System]
            list of associated subsystems
        """
        return list(self.subsystems.values())

    def subsystems_as_tuple(self) -> tuple[System, ...]:
        """output the associated subsystems as tuple

        Returns
        -------
        tuple[System, ...]
            tuple of associated subsystems
        """
        return tuple(self.subsystems.values())

    def subsystems_as_set(self) -> set[System]:
        """output the associated subsystems as set

        Returns
        -------
        set[System]
            set of associated subsystems
        """
        return set(self.subsystems.values())

    def add_supersystem(
        self,
        supersystem: System,
    ) -> None:
        if supersystem.system_id not in self.supersystems:
            self.supersystems[supersystem.system_id] = supersystem
            self.supersystems_ids.add(supersystem.system_id)
            self.supersystems_custom_ids.add(supersystem.custom_identifier)

    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        UserWarning
            if a subsystem is already associated with the given supersystem
        """
        # do not allow adding of subsystems for lowest level systems
        if self._abstraction_level == 0:
            raise RuntimeError(
                (
                    f'Tried to add subsystem to {self}, but it is on the lowest hierarchy '
                    f'level. Systems on the lowest level can not contain other systems.'
                )
            )

        if subsystem.system_id not in self.subsystems:
            self.subsystems[subsystem.system_id] = subsystem
        else:
            raise UserWarning(f'Subsystem {subsystem} was already in supersystem {self}!')

        subsystem.add_supersystem(supersystem=self)

        # register association in corresponding database
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.register_system_association(supersystem=self, subsystem=subsystem)

        # check if a processing station was added
        if isinstance(subsystem, ProcessingStation):
            # set flag
            self._containing_proc_stations = True
            # update property in database
            infstruct_mgr.set_contain_proc_station(system=self)

        loggers.infstrct.info('Successfully added %s to %s.', subsystem, self)

    @overload
    def lowest_level_subsystems(
        self,
        only_processing_stations: Literal[True],
    ) -> tuple[ProcessingStation, ...]: ...

    @overload
    def lowest_level_subsystems(
        self,
        only_processing_stations: Literal[False] = ...,
    ) -> tuple[InfrastructureObject, ...]: ...

    # @lru_cache(maxsize=3)
    def lowest_level_subsystems(
        self,
        only_processing_stations: bool = False,
    ) -> tuple[InfrastructureObject, ...] | tuple[ProcessingStation, ...]:
        """obtain all associated InfrastructureObjects on the lowest hierarchy level

        Parameters
        ----------
        only_processing_stations : bool, optional
            return all associated InfrastructureObjects (False)
            or only ProcessingStations (True), by default False

        Returns
        -------
        tuple[InfrastructureObject, ...]
            tuple with all associated InfrastructureObjects

        Raises
        ------
        RuntimeError
            if system itself is on the lowest hierarchy level
        """

        if self._abstraction_level == 0:
            raise RuntimeError(
                (
                    'Can not obtain lowest level subsystems from '
                    'lowest hierarchy level objects.'
                )
            )

        remaining_abstraction_level = self._abstraction_level - 1
        subsystems = self.subsystems_as_set()

        while remaining_abstraction_level > 0:
            temp: set[System] = set()

            for subsystem in subsystems:
                children = subsystem.subsystems_as_set()
                temp |= children

            subsystems = temp
            remaining_abstraction_level -= 1

        # flatten list and remove duplicates by making a set
        low_lev_subsystems_set = cast(set[InfrastructureObject], set(flatten(subsystems)))
        # filter only processing stations if option chosen
        low_lev_subsystems_lst: list[InfrastructureObject] | list[ProcessingStation]
        if only_processing_stations:
            low_lev_subsystems_lst = filter_processing_stations(
                infstruct_obj_collection=low_lev_subsystems_set
            )
        else:
            low_lev_subsystems_lst = list(low_lev_subsystems_set)

        # sort list by system ID (ascending), so that the order is always the same
        low_lev_subsystems_lst = sorted(
            low_lev_subsystems_lst, key=attrgetter('system_id'), reverse=False
        )

        return tuple(low_lev_subsystems_lst)

    def initialise(self) -> None:
        # assign associated ProcessingStations and corresponding info
        self._assoc_proc_stations = self.lowest_level_subsystems(
            only_processing_stations=True
        )
        self._num_assoc_proc_stations = len(self._assoc_proc_stations)


class ProductionArea(System):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        state: str | None = None,
    ) -> None:
        """Group of processing stations which are considered parallel machines"""

        # initialise base class
        super().__init__(
            env=env,
            system_type=SimSystemTypes.PRODUCTION_AREA,
            custom_identifier=custom_identifier,
            abstraction_level=2,
            name=name,
            state=state,
        )

    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        TypeError
            if a subsystem is not the type this system contains
        """
        # type check: only certain subsystems are allowed for each supersystem
        if not isinstance(subsystem, StationGroup):
            raise TypeError(
                (
                    f'The provided subsystem muste be of type >>StationGroup<<, '
                    f'but it is {type(subsystem)}.'
                )
            )

        super().add_subsystem(subsystem=subsystem)


class StationGroup(System):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        state: str | None = None,
    ) -> None:
        """Group of processing stations which are considered parallel machines"""

        # initialise base class
        super().__init__(
            env=env,
            system_type=SimSystemTypes.STATION_GROUP,
            custom_identifier=custom_identifier,
            abstraction_level=1,
            name=name,
            state=state,
        )

        return None

    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        TypeError
            if a subsystem is not the type this system contains
        """
        # type check: only certain subsystems are allowed for each supersystem
        if not isinstance(subsystem, InfrastructureObject):
            raise TypeError(
                (
                    f'The provided subsystem muste be of type >>InfrastructureObject<<, '
                    f'but it is {type(subsystem)}.'
                )
            )

        super().add_subsystem(subsystem=subsystem)


# INFRASTRUCTURE COMPONENTS


class InfrastructureObject(System, metaclass=ABCMeta):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
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
        env: simulation environment in which the infrastructure object is embedded
        custom_identifier: unique user-defined custom ID of the given object \
            necessary for user interfaces
        capacity: capacity of the infrastructure object, if multiple processing \
            slots available at the same time > 1, default=1
        """
        # [HIERARCHICAL SYSTEM INFORMATION]
        # contrary to other system types no bucket because a processing station
        # is the smallest unit in the system view/analysis
        # initialise base class >>System<<
        # calls to Infrastructure Manager to register object
        super().__init__(
            env=env,
            system_type=SimSystemTypes.RESOURCE,
            custom_identifier=custom_identifier,
            abstraction_level=0,
            name=name,
            state=state,
        )
        self.capacity = capacity
        self.res_type: str
        # [STATS] Monitoring
        self._stat_monitor = monitors.InfStructMonitor(
            env=env,
            obj=self,
            init_state=state,
            possible_states=possible_states,
        )
        # [SALABIM COMPONENT]
        self._sim_control = SimulationComponent(
            env=env,
            name=self.name,
            pre_process=self.pre_process,
            sim_logic=self.sim_logic,
            post_process=self.post_process,
        )
        # [LOGIC] logic queue
        # each resource uses one associated logic queue, logic queues are not
        # physically available
        queue_name: str = f'queue_{self.name}'
        self.logic_queue = salabim.Queue(name=queue_name, env=self.env)
        # currently available jobs on that resource
        self.contents: dict[LoadID, Job] = {}
        # [STATS] additional information
        # number of inputs/outputs
        self.num_inputs: int = 0
        self.num_outputs: int = 0

        # time characteristics
        self._proc_time: Timedelta = Timedelta.min
        # setup time: if a setup time is provided use always this time and
        # ignore job-related setup times
        self._setup_time = setup_time
        self._use_const_setup_time: bool
        if self._setup_time is not None:
            self._use_const_setup_time = True
        else:
            self._use_const_setup_time = False

    # override for corresponding classes
    @property
    def stat_monitor(self) -> monitors.InfStructMonitor:
        return self._stat_monitor

    @property
    def sim_control(self) -> SimulationComponent:
        return self._sim_control

    @property
    def use_const_setup_time(self) -> bool:
        return self._use_const_setup_time

    @property
    def proc_time(self) -> Timedelta:
        return self._proc_time

    @proc_time.setter
    def proc_time(
        self,
        new_proc_time: Timedelta,
    ) -> None:
        if isinstance(new_proc_time, Timedelta):
            self._proc_time = new_proc_time
        else:
            raise TypeError(
                (
                    f'The processing time must be of type >>Timedelta<<, '
                    f'but it is >>{type(new_proc_time)}<<'
                )
            )

    @property
    def setup_time(self) -> Timedelta | None:
        return self._setup_time

    @setup_time.setter
    def setup_time(
        self,
        new_setup_time: Timedelta,
    ) -> None:
        if self._use_const_setup_time:
            raise RuntimeError(
                (
                    f'Tried to change setup time of >>{self}<<, but it is '
                    f'configured to use a constant time of >>{self._setup_time}<<'
                )
            )

        if isinstance(new_setup_time, Timedelta):
            self._setup_time = new_setup_time
        else:
            raise TypeError(
                (
                    f'The setup time must be of type >>Timedelta<<, '
                    f'but it is >>{type(new_setup_time)}<<'
                )
            )

    def add_content(
        self,
        job: Job,
    ) -> None:
        """add contents to the InfrastructureObject"""
        job_id = job.job_id
        if job_id not in self.contents:
            self.contents[job_id] = job
        else:
            raise KeyError(f'Job {job} already in contents of {self}')

    def remove_content(
        self,
        job: Job,
    ) -> None:
        """remove contents from the InfrastructureObject"""
        job_id = job.job_id
        if job_id in self.contents:
            del self.contents[job_id]
        else:
            raise KeyError(f'Job {job} not in contents of {self}')

    def put_job(
        self,
        job: Job,
    ) -> Generator[Any, None, InfrastructureObject]:
        # ALLOCATION REQUEST
        # TODO ++++++++++ add later ++++++++++++
        # time component: given start date of operation
        # returning release date, waiting for release date or release early
        dispatcher = self.env.dispatcher
        infstruct_mgr = self.env.infstruct_mgr
        # call dispatcher to check for allocation rule
        # resets current feasibility status
        yield self.sim_control.hold(0)
        is_agent, alloc_agent = dispatcher.check_alloc_dispatch(job=job)
        target_station: InfrastructureObject
        if is_agent and alloc_agent is None:
            raise ValueError('Agent is set, but no agent is provided.')
        elif is_agent and alloc_agent is not None:
            # if agent is set, set flags and calculate feature vector
            # as long as there is no feasible action
            while not alloc_agent.action_feasible:
                # ** SET external Gym flag, build feature vector
                dispatcher.request_agent_alloc(job=job)
                # ** Break external loop
                # ** [only step] Calc reward in Gym-Env
                loggers.agents.debug(
                    (
                        '--------------- DEBUG: call before hold(0) at %s, %s',
                        self.env.t(),
                        self.env.t_as_dt(),
                    )
                )
                yield self.sim_control.hold(0)
                # ** make and set decision in Gym-Env --> RESET external Gym flag
                loggers.agents.debug(
                    ('--------------- DEBUG: call after hold(0) at %s', self.env.t())
                )
                loggers.agents.debug(
                    'Action feasibility: current %s, past %s',
                    alloc_agent.action_feasible,
                    alloc_agent.past_action_feasible,
                )

                # obtain target station, check for feasibility
                # --> SET ``agent.action_feasible``
                target_station = dispatcher.request_job_allocation(job=job, is_agent=is_agent)
                # historic value for reward calculation,
                # prevent overwrite from ``check_alloc_dispatch``
                alloc_agent.past_action_feasible = alloc_agent.action_feasible
        else:
            # simply obtain target station if no agent decision is needed
            target_station = dispatcher.request_job_allocation(job=job, is_agent=is_agent)

        # TODO check removal
        # yield self.sim_control.hold(0)

        # get logic queue
        logic_queue = target_station.logic_queue
        # check if the target is a sink
        if isinstance(target_station, Sink):
            pass
        elif isinstance(target_station, ProcessingStation):
            # check if associated buffers exist
            loggers.prod_stations.debug('[%s] Check for buffers', self)
            buffers = target_station.buffers

            if buffers:
                # [STATE:InfrStructObj] BLOCKED
                infstruct_mgr.update_res_state(obj=self, state='BLOCKED')
                # [STATE:Job] BLOCKED
                dispatcher.update_job_state(job=job, state='BLOCKED')
                yield self.sim_control.to_store(
                    store=target_station.stores,
                    item=job,
                    fail_delay=FAIL_DELAY,
                    fail_priority=1,
                )
                if self.sim_control.failed():
                    raise UserWarning(
                        (
                            f'Store placement failed after {FAIL_DELAY} time steps. '
                            f'There seems to be deadlock.'
                        )
                    )
                # [STATE:Buffer] trigger state setting for target buffer
                salabim_store = self.sim_control.to_store_store()
                if salabim_store is None:
                    raise ValueError('No store object honoured.')
                buffer = target_station.buffer_by_store_name(salabim_store.name())
                buffer.sim_control.activate()
                # [CONTENT:Buffer] add content
                buffer.add_content(job=job)
                # [STATS:Buffer] count number of inputs
                buffer.num_inputs += 1
                loggers.prod_stations.debug(
                    'obj = %s \t type of buffer >>%s<< = %s at %s',
                    self,
                    buffer,
                    type(buffer),
                    self.env.now(),
                )
            else:
                # adding request to machine
                # currently not possible because machines are components,
                # but resources which could be requested are not
                pass

        # [Job] enter logic queue after physical placement
        job.enter(logic_queue)
        # [STATS:WIP] REMOVING WIP FROM CURRENT STATION
        # remove only if it was added before, only case if the last operation exists
        if job.last_op is not None:
            self.stat_monitor.change_WIP(job=job, remove=True)
        # [STATS:WIP] ADDING WIP TO TARGET STATION
        # add only if there is a next operation, only case if the current operation exists
        if job.current_op is not None:
            target_station.stat_monitor.change_WIP(job=job, remove=False)

        # activate target processing station if passive
        if target_station.sim_control.ispassive():
            target_station.sim_control.activate()

        loggers.prod_stations.debug('[%s] Put Job %s in queue %s', self, job, logic_queue)

        # [STATE:InfrStructObj] WAITING
        infstruct_mgr.update_res_state(obj=self, state='WAITING')
        # [STATE:Job] successfully placed --> WAITING
        dispatcher.update_job_state(job=job, state='WAITING')
        # [STATS:InfrStructObj] count number of outputs
        self.num_outputs += 1

        return target_station

    def get_job(self) -> Generator[Any, Any, Job]:
        """
        getting jobs from associated predecessor resources
        """
        # entering target machine (logic_buffer)
        ## logic queue: job queue regardless of physical buffers
        ### entity physically on machine, but no true holding resource object
        ### (violates load-resource model)
        ### no capacity restrictions between resources, e.g.,
        ### source can endlessly produce entities
        ## --- logic ---
        ## job enters logic queue of machine with unrestricted capacity
        ## each machine can have an associated physical buffer
        dispatcher = self.env.dispatcher
        infstruct_mgr = self.env.infstruct_mgr
        # request job and its time characteristics from associated queue
        # TODO retrieve times from job object directly
        job, job_proc_time, job_setup_time = dispatcher.request_job_sequencing(req_obj=self)

        ### UPDATE JOB PROCESS INFO IN REQUEST FUNCTION???

        # update time characteristics of the infrastructure object
        # contains additional checks if the target values are allowed
        self.proc_time = job_proc_time
        if job_setup_time is not None:
            loggers.prod_stations.debug(
                (
                    '[SETUP TIME DETECTED] job ID %s at %s on machine ID %s '
                    'with setup time %s'
                ),
                job.job_id,
                self.env.now(),
                self.custom_identifier,
                self.setup_time,
            )
            self.setup_time = job_setup_time

        # Processing Station only
        # request and get job from associated buffer if it exists
        if isinstance(self, ProcessingStation) and self._buffers:
            yield self.sim_control.from_store(
                store=self.stores, filter=lambda item: item.job_id == job.job_id
            )
            salabim_store = self.sim_control.from_store_store()
            if salabim_store is None:
                raise ValueError('No store object honoured.')
            buffer = self.buffer_by_store_name(salabim_store.name())
            # [STATS:Buffer] count number of outputs
            buffer.num_outputs += 1
            # [CONTENT:Buffer] remove content
            buffer.remove_content(job=job)
            # [STATE:Buffer] trigger state setting for target buffer
            buffer.sim_control.activate()

        # RELEVANT INFORMATION BEFORE PROCESSING
        dispatcher.update_job_process_info(job=job, preprocess=True)
        # [STATS] count number of inputs
        self.num_inputs += 1
        # [CONTENT] add content
        self.add_content(job=job)

        # SETUP
        if self.setup_time is not None:
            # [STATE:InfrStructObj]
            infstruct_mgr.update_res_state(obj=self, state='SETUP')
            # [STATE:Job]
            dispatcher.update_job_state(job=job, state='SETUP')
            loggers.prod_stations.debug(
                '[START SETUP] job ID %s at %s on machine ID %s with setup time %s',
                job.job_id,
                self.env.now(),
                self.custom_identifier,
                self.setup_time,
            )
            sim_time = self.env.td_to_simtime(timedelta=self.setup_time)
            yield self.sim_control.hold(sim_time)

        # [STATE:InfrStructObj] PROCESSING
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')
        # [STATE:Job] PROCESSING
        dispatcher.update_job_state(job=job, state='PROCESSING')

        return job

    ### PROCESS LOGIC
    # each method of 'pre_process', 'sim_control', 'post_process' must be
    # implemented in the child classes
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

    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by
        the environment's "finalise_sim" method
        """
        infstruct_mgr = self.env.infstruct_mgr
        # set finish state for each infrastructure object no matter of which child class
        infstruct_mgr.update_res_state(obj=self, state='FINISH')
        # finalise stat gathering
        self._stat_monitor.finalise_stats()


class StorageLike(InfrastructureObject):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: int | Infinite = INF,
        state: str = 'INIT',
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
        fill_level_init: int = 0,
    ) -> None:
        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
        )

        self.fill_level_init = fill_level_init
        self._stat_monitor = monitors.StorageMonitor(
            env=env,
            obj=self,
            init_state=state,
            possible_states=possible_states,
        )

        self._sim_control = StorageComponent(
            env=env,
            name=self.name,
            capacity=capacity,
            pre_process=self.pre_process,
            sim_logic=self.sim_logic,
            post_process=self.post_process,
        )

    @property
    def stat_monitor(self) -> monitors.StorageMonitor:
        return self._stat_monitor

    @property
    def sim_control(self) -> StorageComponent:
        return self._sim_control

    @property
    def fill_level(self) -> int:
        return len(self.sim_control.store)


class ProcessingStation(InfrastructureObject):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
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
        buffers: Iterable[Buffer] | None = None,
    ) -> None:
        """
        env: simulation environment in which the infrastructure object is embedded
        capacity: capacity of the infrastructure object, if multiple processing \
            slots available at the same time > 1, default=1
        """
        # initialise base class
        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
        )

        # add physical buffers, more than one allowed
        # contrary to logic queues buffers are infrastructure objects and exist physically
        self._buffers: set[Buffer]
        if buffers is None:
            self._buffers = set()
        else:
            self._buffers = set(buffers).copy()

        # add processing station to the associated ones of each buffer
        # necessary because if the number of resources for one buffer exceeds its capacity
        # deadlocks are possible
        for buffer in self._buffers:
            buffer.add_prod_station(prod_station=self)
        # Salabim Store objects for retrieval
        self._stores = self._get_stores()
        self._stores_map = self._map_store_name_to_buffer()

    @property
    def buffers(self) -> set[Buffer]:
        return self._buffers

    @property
    def stores(self) -> set[salabim.Store]:
        return self._stores

    @property
    def stores_map(self) -> dict[str, Buffer]:
        return self._stores_map

    def _get_stores(self) -> set[salabim.Store]:
        return {buffer.sim_control.store for buffer in self._buffers}

    def _map_store_name_to_buffer(self) -> dict[str, Buffer]:
        return {buffer.sim_control.store_name: buffer for buffer in self._buffers}

    def buffer_by_store_name(
        self,
        store_name: str,
    ) -> Buffer:
        buffer = self._stores_map.get(store_name, None)
        if buffer is None:
            raise KeyError(f'No buffer with name {store_name} found.')
        return buffer

    def buffers_as_tuple(self) -> tuple[Buffer, ...]:
        return tuple(self._buffers)

    # TODO: add station group information or delete
    """
    @property
    def station_group_id(self) -> SystemID:
        return self._station_group_id
    
    @property
    def station_group(self) -> StationGroup:
        return self._station_group
    """

    def add_buffer(
        self,
        buffer: Buffer,
    ) -> None:
        """
        adding buffer to the current associated ones
        """
        # only buffer types allowed
        if not isinstance(buffer, Buffer):
            raise TypeError(
                (
                    'Object is no Buffer type. Only objects '
                    'of type Buffer can be added as buffers.'
                )
            )
        # check if already present
        if buffer not in self._buffers:
            self._buffers.add(buffer)
            self._stores = self._get_stores()
            self._stores_map = self._map_store_name_to_buffer()
            buffer.add_prod_station(prod_station=self)
        else:
            loggers.prod_stations.warning(
                (
                    'The Buffer >>%s<< is already associated with the resource >>%s<<. '
                    'Buffer was not added to the resource.'
                ),
                buffer,
                self,
            )

    def remove_buffer(
        self,
        buffer: Buffer,
    ) -> None:
        """
        removing buffer from the current associated ones
        """
        if buffer in self._buffers:
            self._buffers.remove(buffer)
            self._stores = self._get_stores()
            self._stores_map = self._map_store_name_to_buffer()
            buffer.remove_prod_station(prod_station=self)
        else:
            raise KeyError(
                (
                    f'The buffer >>{buffer}<< is not associated with the resource '
                    f'>>{self}<< and therefore could not be removed.'
                )
            )

    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='WAITING')

    def sim_logic(self) -> Generator[Any, None, None]:
        dispatcher = self.env.dispatcher
        while True:
            # initialise state by passivating machines
            # resources are activated by other resources
            if len(self.logic_queue) == 0:
                yield self.sim_control.passivate()
            loggers.prod_stations.debug('[MACHINE: %s] is getting job from queue', self)

            # ONLY PROCESSING STATIONS ARE ASKING FOR SEQUENCING
            job = yield from self.get_job()

            loggers.prod_stations.debug(
                '[START] job ID %s at %s on machine ID %s with proc time %s',
                job.job_id,
                self.env.now(),
                self.custom_identifier,
                self.proc_time,
            )
            # PROCESSING
            sim_time = self.env.td_to_simtime(timedelta=self.proc_time)
            yield self.sim_control.hold(sim_time)
            dispatcher.update_job_process_info(job=job, preprocess=False)
            loggers.prod_stations.debug(
                '[END] job ID %s at %s on machine ID %s',
                job.job_id,
                self.env.now(),
                self.custom_identifier,
            )

            _ = yield from self.put_job(job=job)
            # [CONTENT:ProdStation] remove content
            self.remove_content(job=job)

    def post_process(self) -> None:
        pass

    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by
        the environment's "finalise_sim" method
        """
        # each resource object class has dedicated finalise methods which
        # must be called by children
        super().finalise()


class Machine(ProcessingStation):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
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
        buffers: Iterable[Buffer] | None = None,
    ) -> None:
        """
        ADD LATER
        """
        # assign object information
        self.res_type = 'Machine'

        # initialise base class
        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
            buffers=buffers,
        )


class Buffer(StorageLike):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: int | Infinite = INF,
        state: str = 'INIT',
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
        fill_level_init: int = 0,
    ) -> None:
        """
        capacity: capacity of the buffer, can be infinite
        """
        self.res_type = 'Buffer'
        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
            fill_level_init=fill_level_init,
        )
        # material flow relationships
        self._associated_prod_stations: set[ProcessingStation] = set()
        self._count_associated_prod_stations: int = 0

    @property
    def level_db(self) -> DataFrame:
        return self._stat_monitor.level_db

    @property
    def wei_avg_fill_level(self) -> float | None:
        return self._stat_monitor.wei_avg_fill_level

    def add_prod_station(
        self,
        prod_station: ProcessingStation,
    ) -> None:
        """
        function to add processing stations which are associated with
        """
        if not isinstance(prod_station, ProcessingStation):
            raise TypeError(
                (
                    'Object is no ProcessingStation type. Only objects of type '
                    '>>ProcessingStation<< can be added to a buffer.'
                )
            )

        # check if adding a new resource exceeds the given capacity
        # each associated processing station needs one storage place in the buffer
        # else deadlocks are possible
        if (self._count_associated_prod_stations + 1) > self.capacity:
            raise UserWarning(
                (
                    f'Tried to add a new resource to buffer {self}, '
                    f'but the number of associated resources exceeds '
                    f'its capacity which could result in deadlocks.'
                )
            )
        # check if processing station can be added
        if prod_station not in self._associated_prod_stations:
            self._associated_prod_stations.add(prod_station)
            self._count_associated_prod_stations += 1
        else:
            loggers.buffers.warning(
                (
                    'The Processing Station >>%s<< is '
                    'already associated with the resource >>%s<<. '
                    'Processing Station was not added to the resource.'
                ),
                prod_station,
                self,
            )

    def remove_prod_station(
        self,
        prod_station: ProcessingStation,
    ) -> None:
        """
        removing a processing station from the current associated ones
        """
        if prod_station in self._associated_prod_stations:
            self._associated_prod_stations.remove(prod_station)
            self._count_associated_prod_stations -= 1
        else:
            raise KeyError(
                (
                    f'The processing station >>{prod_station}<< is not '
                    f'associated with the resource >>{self}<< '
                    f'and therefore could not be removed.'
                )
            )

    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='EMPTY')

    def sim_logic(self) -> Generator[None, None, None]:
        infstruct_mgr = self.env.infstruct_mgr
        while True:
            loggers.prod_stations.debug('[BUFFER: %s] Invoking at %s', self, self.env.now())
            # full
            if self.sim_control.store.available_quantity() == 0:
                # [STATE] FULL
                infstruct_mgr.update_res_state(obj=self, state='FULL')
                loggers.prod_stations.debug(
                    '[BUFFER: %s] Set to >>FULL<< at %s',
                    self,
                    self.env.now(),
                )
            # empty
            elif (
                self.sim_control.store.available_quantity()
                == self.sim_control.store.capacity()
            ):
                # [STATE] EMPTY
                infstruct_mgr.update_res_state(obj=self, state='EMPTY')
                loggers.prod_stations.debug(
                    '[BUFFER: %s] Set to >>EMPTY<< at %s',
                    self,
                    self.env.now(),
                )
            else:
                # [STATE] INTERMEDIATE
                infstruct_mgr.update_res_state(obj=self, state='INTERMEDIATE')
                loggers.prod_stations.debug(
                    '[BUFFER: %s] Neither >>EMPTY<< nor >>FULL<< at %s',
                    self,
                    self.env.now(),
                )

            yield self.sim_control.passivate()

    def post_process(self) -> None:
        pass

    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by
        the environment's "finalise_sim" method
        """
        # each resource object class has dedicated finalise methods which
        # must be called by children
        super().finalise()


class Source(InfrastructureObject):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
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
        proc_time: Timedelta = Timedelta(hours=2),
        job_generator: RandomJobGenerator | None = None,
        job_sequence: Iterator[tuple[SystemID, SystemID, OrderTime]] | None = None,
        num_gen_jobs: int | None = None,
    ) -> None:
        """
        num_gen_jobs: total number of jobs to be generated
        """
        # assign object information and register object in the environment
        self.res_type = 'Source'

        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
        )

        # generation method
        if job_generator is None and job_sequence is None:
            raise ValueError(
                'Random job generator or job sequence necessary for job generation'
            )
        elif job_generator is not None and job_sequence is not None:
            raise ValueError('Only one job generation method allowed')

        self.job_generator = job_generator
        self.job_sequence = job_sequence

        # TODO REWORK
        # initialise component with necessary process function
        random.seed(42)

        # parameters
        self.proc_time = proc_time
        # indicator if an time equivalent should be used
        self.use_stop_time: bool = False
        self.num_gen_jobs: int | None = None
        if num_gen_jobs is not None:
            self.num_gen_jobs = num_gen_jobs
        else:
            self.use_stop_time = True

        # triggers and flags
        self.stop_job_gen_cond_reg: bool = False
        self.stop_job_gen_state = salabim.State('stop_job_gen', env=self.env)

    def _obtain_proc_time(self) -> float:
        """
        function to generate a constant or random processing time
        """
        proc_time = self.env.td_to_simtime(timedelta=self.proc_time)
        return proc_time

    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')

        # check if ConditionSetter is registered if needed
        if self.use_stop_time and not self.stop_job_gen_cond_reg:
            raise ValueError(
                (
                    f'[SOURCE {self}]: Stop time condition should be used, '
                    f'but no ConditionSetter is registered.'
                )
            )

    def sim_logic(self) -> Generator[None, None, None]:
        # counter for debugging, else endless generation
        count = 0
        infstruct_mgr = self.env.infstruct_mgr
        dispatcher = self.env.dispatcher

        # use machine custom identifiers for generation
        # machines = infstruct_mgr.res_db.loc[infstruct_mgr.res_db['res_type']=='Machine']
        # machines_custom_ids = machines['custom_id'].to_list()

        # use station group custom identifiers for generation
        # station_groups_custom_ids = infstruct_mgr.station_group_db['custom_id'].to_list()

        # ?? only for random generation?
        """
        # use production area custom identifiers for generation
        prod_areas = infstruct_mgr.prod_area_db.copy()
        # prod_area_custom_ids = 
        # prod_areas.loc[prod_areas['containing_proc_stations'] == True,'custom_id'].to_list()
        prod_area_custom_ids = 
        prod_areas.loc[prod_areas['containing_proc_stations'] == True,'custom_id']
        #prod_area_system_ids = 
        # prod_areas.loc[prod_areas['containing_proc_stations'] == True,:].index.to_list()
        # get station group custom identifiers which are associated with 
        # the relevant production areas
        stat_groups = infstruct_mgr.station_group_db.copy()
        stat_group_ids: dict[CustomID, list[CustomID]] = {}
        for PA_sys_id, PA_custom_id in prod_area_custom_ids.items():
            # get associated station group custom IDs by their 
            # corresponding production area system ID
            candidates = 
            stat_groups.loc[(stat_groups['prod_area_id'] == PA_sys_id), 'custom_id'].to_list()
            # map production area custom ID to the associated station group custom IDs
            stat_group_ids[PA_custom_id] = candidates
            
        loggers.sources.debug(f"[SOURCE: {self}] Stat Group IDs: {stat_group_ids}")
        """

        # ** new: job generation by sequence
        # !! currently only one production area
        if self.job_sequence is None:
            raise ValueError('Job sequence is not set')
        for prod_area_id, station_group_id, order_times in self.job_sequence:
            if not self.use_stop_time:
                # use number of generated jobs as stopping criterion
                if count == self.num_gen_jobs:
                    break
            else:
                # stop if stopping time is reached
                # flag set by corresponding ConditionSetter
                if self.stop_job_gen_state.get():
                    break

            # start at t=0 with generation
            # generate object
            ## random job properties
            ## currently: each job passes each machine, only one machine of
            # each operation type
            # mat_ProcTimes, mat_JobMachID =
            # self.job_generator.gen_rnd_job(n_machines=self.env.num_proc_stations)
            # job = Job(dispatcher=dispatcher, proc_times=mat_ProcTimes.tolist(),
            #          machine_order=mat_JobMachID.tolist())
            # mat_ProcTimes, mat_JobMachID =
            # self.job_generator.gen_rnd_job_by_ids(ids=machines_custom_ids)
            # mat_ProcTimes, mat_JobExOrder =
            # self.job_generator.gen_rnd_job_by_ids(
            # ids=station_groups_custom_ids, min_proc_time=5)
            """
            (job_ex_order, job_target_station_groups, 
             proc_times, setup_times) = self.job_generator.gen_rnd_job_by_ids(
                exec_system_ids=prod_area_custom_ids,
                target_station_group_ids=stat_group_ids,
                min_proc_time=5,
                gen_setup_times=True,
            )
            loggers.sources.debug(f"[SOURCE: {self}] 
            ProcTimes {proc_times} at {self.env.now()}")
            """

            # assign random priority
            # prio = self.job_generator.gen_prio() + count
            prio = count
            # prio = [2,8]
            # assign starting and ending dates
            start_date_init = Datetime(2023, 11, 20, hour=6, tzinfo=TIMEZONE_UTC)
            end_date_init = Datetime(2023, 12, 1, hour=10, tzinfo=TIMEZONE_UTC)

            # loggers.sources.debug('[SOURCE: %s] Exec Order: %s', self, job_ex_order)
            # loggers.sources.debug('[SOURCE: %s] %s', self, job_target_station_groups)
            # !! job init with CustomID, but SystemID used
            # TODO: change initialisation to SystemID
            """
            job = Job(dispatcher=dispatcher,
                      exec_systems_order=job_ex_order,
                      target_stations_order=job_target_station_groups,
                      proc_times=proc_times,
                      setup_times=setup_times,
                      prio=prio,
                      planned_starting_date=start_date_init,
                      planned_ending_date=end_date_init)
            """
            job = Job(
                dispatcher=dispatcher,
                exec_systems_order=[prod_area_id],
                target_stations_order=[station_group_id],
                proc_times=[order_times.proc],
                setup_times=[order_times.setup],
                prio=prio,
                planned_starting_date=start_date_init,
                planned_ending_date=end_date_init,
            )
            loggers.sources.debug(
                '[SOURCE: %s] Job target station group: %s',
                self,
                job.operations[0].target_station_group,
            )
            # [Call:DISPATCHER]
            dispatcher.release_job(job=job)
            # [STATS:Source] count number of inputs
            # (source: generation of jobs or entry in pipeline)
            # implemented in 'get_job' method which is not executed by source objects
            self.num_inputs += 1
            loggers.sources.debug(
                '[SOURCE: %s] Generated %s at %s', self, job, self.env.now()
            )

            loggers.sources.debug('[SOURCE: %s] Request allocation...', self)
            # put job via 'put_job' function,
            # implemented in parent class 'InfrastructureObject'
            target_proc_station = yield from self.put_job(job=job)
            loggers.sources.debug(
                '[SOURCE: %s] PUT JOB with ret = %s', self, target_proc_station
            )
            # [STATE:Source] put in 'WAITING' by 'put_job' method but still processing
            # only 'WAITING' if all jobs are generated
            infstruct_mgr.update_res_state(obj=self, state='PROCESSING')

            # hold for defined generation time (constant or statistically distributed)
            # if hold time elapsed start new generation
            proc_time = self._obtain_proc_time()
            loggers.sources.debug(
                '[SOURCE: %s] Hold for >>%s<< at %s', self, proc_time, self.env.now()
            )

            yield self.sim_control.hold(proc_time)
            # set counter up
            count += 1

        # [STATE:Source] WAITING
        infstruct_mgr.update_res_state(obj=self, state='WAITING')

    def post_process(self) -> None:
        pass


class Sink(InfrastructureObject):
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
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
        num_gen_jobs: total number of jobs to be generated
        """
        self.res_type = 'Sink'
        super().__init__(
            env=env,
            custom_identifier=custom_identifier,
            name=name,
            setup_time=setup_time,
            capacity=capacity,
            state=state,
            possible_states=possible_states,
        )

    def pre_process(self) -> None:
        # currently sinks are 'PROCESSING' the whole time
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')

    def sim_logic(self) -> Generator[None, None, None]:
        dispatcher = self.env.dispatcher
        while True:
            if len(self.logic_queue) == 0:
                yield self.sim_control.passivate()
            loggers.sinks.debug('[SINK: %s] is getting job from queue', self)
            job = cast(Job, self.logic_queue.pop())
            # [Call:DISPATCHER] data collection: finalise job
            dispatcher.finish_job(job=job)
            # ?? destroy job object?
            # if job object destroyed, unsaved information is lost
            # if not destroyed memory usage could increase
            # TODO write finalised job information to database (disk)

    def post_process(self) -> None:
        pass


# ** load components


class Operation:
    def __init__(
        self,
        dispatcher: Dispatcher,
        job: Job,
        exec_system_identifier: SystemID,
        proc_time: Timedelta,
        setup_time: Timedelta | None = None,
        target_station_group_identifier: SystemID | None = None,
        prio: int | None = None,
        planned_starting_date: Datetime | None = None,
        planned_ending_date: Datetime | None = None,
        custom_identifier: CustomID | None = None,
        state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'WAITING',
            'SETUP',
            'PROCESSING',
            'BLOCKED',
            'FAILED',
            'PAUSED',
        ),
    ) -> None:
        """
        ADD DESCRIPTION
        """
        # TODO: change to OrderTime object
        # !! perhaps processing times in future multiple entries depending on
        # !! associated machine
        # change of input format necessary, currently only one machine for each operation
        # no differing processing times for different machines or groups

        # assign operation information
        self._dispatcher = dispatcher
        self._job = job
        self._job_id = job.job_id
        self._exec_system_identifier = exec_system_identifier
        self._target_station_group_identifier = target_station_group_identifier

        # [STATS] Monitoring
        self._stat_monitor = monitors.Monitor(
            env=self._dispatcher.env,
            obj=self,
            init_state=state,
            possible_states=possible_states,
        )

        # process information
        # time characteristics
        self.proc_time = proc_time
        self.setup_time = setup_time
        if self.setup_time is not None:
            self.order_time = self.proc_time + self.setup_time
        else:
            self.order_time = self.proc_time
        # inter-process time characteristics
        # time of release
        self.time_release = DEFAULT_DATETIME
        # time of first operation starting point
        self.time_actual_starting = DEFAULT_DATETIME
        # starting date deviation
        self.starting_date_deviation: Timedelta | None = None
        # time of last operation ending point
        self.time_actual_ending = DEFAULT_DATETIME
        # ending date deviation
        self.ending_date_deviation: Timedelta | None = None
        # lead time
        self.lead_time = Timedelta()
        # starting and end dates
        # validate time zone information for given datetime objects
        if planned_starting_date is not None:
            _dt_mgr.validate_dt_UTC(planned_starting_date)
        self.time_planned_starting = planned_starting_date
        if planned_ending_date is not None:
            _dt_mgr.validate_dt_UTC(planned_ending_date)
        self.time_planned_ending = planned_ending_date
        # in future setting starting points in advance possible
        self.is_finished: bool = False
        self.is_released: bool = False
        # priority, default: -1 --> no prio set
        self._prio = prio

        ########### adding machine instances
        ### perhaps adding machine sets if multiple machines possible (machine groups)
        # assignment of machine instance by dispatcher
        # from dispatcher: op_id, name, target_machine
        # register operation instance
        current_state = self._stat_monitor.get_current_state()

        # registration: only return OpID, other properties directly
        # written by dispatcher method
        # add target station group by station group identifier
        self.target_exec_system: System | None = None
        self.target_station_group: StationGroup | None = None
        self.time_creation: Datetime | None = None

        self._op_id = self.dispatcher.register_operation(
            op=self,
            exec_system_identifier=self._exec_system_identifier,
            target_station_group_identifier=target_station_group_identifier,
            custom_identifier=custom_identifier,
            state=current_state,
        )

    def __repr__(self) -> str:
        return (
            f'Operation(ProcTime: {self.proc_time}, '
            f'ExecutionSystemID: {self._exec_system_identifier}, '
            f'SGI: {self._target_station_group_identifier})'
        )

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    @property
    def stat_monitor(self) -> monitors.Monitor:
        return self._stat_monitor

    @property
    def op_id(self) -> LoadID:
        return self._op_id

    @property
    def job(self) -> Job:
        return self._job

    @property
    def job_id(self) -> LoadID:
        return self._job_id

    @property
    def exec_system_identifier(self) -> SystemID:
        return self._exec_system_identifier

    @property
    def target_station_group_identifier(self) -> SystemID | None:
        return self._target_station_group_identifier

    @property
    def prio(self) -> int | None:
        return self._prio

    @prio.setter
    def prio(
        self,
        new_prio: int,
    ) -> None:
        """setting of priority
        prio can be initialized as None,
        but every change has to be an integer value

        Parameters
        ----------
        new_prio : int
            new priority which should be set

        Raises
        ------
        TypeError
            if other types are provided
        """
        if not isinstance(new_prio, int):
            raise TypeError(
                f'The type of {new_prio} must be >>int<<, but it is {type(new_prio)}'
            )
        else:
            self._prio = new_prio
            # REWORK changing OP prio must change job prio but only if op is job's current one


class Job(salabim.Component):
    def __init__(
        self,
        dispatcher: Dispatcher,
        exec_systems_order: Sequence[SystemID],
        proc_times: Sequence[Timedelta],
        target_stations_order: Sequence[SystemID | None] | None = None,
        setup_times: Sequence[Timedelta | None] | None = None,
        prio: int | Sequence[int | None] | None = None,
        planned_starting_date: Datetime | Sequence[Datetime | None] | None = None,
        planned_ending_date: Datetime | Sequence[Datetime | None] | None = None,
        custom_identifier: CustomID | None = None,
        state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'WAITING',
            'SETUP',
            'PROCESSING',
            'BLOCKED',
            'FAILED',
            'PAUSED',
        ),
        additional_info: dict[str, CustomID] | None = None,
        **kwargs,
    ) -> None:
        """
        ADD DESCRIPTION
        """
        # add not provided information
        # target station identifiers
        # if target_stations_order is None:
        op_target_stations: Sequence[SystemID | None]
        if isinstance(target_stations_order, Sequence):
            op_target_stations = target_stations_order
        else:
            op_target_stations = [None] * len(exec_systems_order)
        # setup times
        if setup_times is None:
            setup_times = [None] * len(proc_times)

        # prio
        self.op_wise_prio: bool
        self._prio: int | None
        op_prios: Sequence[int | None]
        if isinstance(prio, Sequence):
            op_prios = prio
            # job prio later set by 'get_next_operation' method
            self.op_wise_prio = True
            self._prio = None
        else:
            # job priority applies to all operations
            # priority, default: None --> no prio set
            op_prios = [None] * len(proc_times)
            # set job priority as a whole
            self._prio = prio
            self.op_wise_prio = False

        # planned dates
        self.op_wise_starting_date: bool
        self.op_wise_ending_date: bool
        self.time_planned_starting: Datetime | None
        self.time_planned_ending: Datetime | None
        op_starting_dates: Sequence[Datetime | None]
        op_ending_dates: Sequence[Datetime | None]
        if isinstance(planned_starting_date, Sequence):
            # operation-wise defined starting dates
            # datetime validation done in operation class
            op_starting_dates = planned_starting_date
            self.time_planned_starting = None
            # job starting date later set by 'get_next_operation' method
            self.op_wise_starting_date = True
        else:
            # only job-wise defined starting date
            op_starting_dates = [None] * len(proc_times)
            # validate time zone information for given datetime object
            if planned_starting_date is not None:
                _dt_mgr.validate_dt_UTC(planned_starting_date)
            self.time_planned_starting = planned_starting_date
            self.op_wise_starting_date = False
        if isinstance(planned_ending_date, Sequence):
            # operation-wise defined ending dates
            # datetime validation done in operation class
            op_ending_dates = planned_ending_date
            self.time_planned_ending = None
            # job ending date later set by 'get_next_operation' method
            self.op_wise_ending_date = True
        else:
            # only job-wise defined starting date
            op_ending_dates = [None] * len(proc_times)
            # validate time zone information for given datetime object
            if planned_ending_date is not None:
                _dt_mgr.validate_dt_UTC(planned_ending_date)
            self.time_planned_ending = planned_ending_date
            self.op_wise_ending_date = False

        ### VALIDITY CHECK ###
        # length of provided identifiers and lists must match
        if target_stations_order is not None:
            if len(target_stations_order) != len(exec_systems_order):
                raise ValueError(
                    (
                        'The number of target stations must match '
                        'the number of execution systems.'
                    )
                )
        if len(proc_times) != len(exec_systems_order):
            raise ValueError(
                (
                    'The number of processing times must match '
                    'the number of execution systems.'
                )
            )
        if len(setup_times) != len(proc_times):
            raise ValueError(
                'The number of setup times must match the number of processing times.'
            )
        if len(op_prios) != len(proc_times):
            raise ValueError(
                (
                    'The number of operation priorities must match '
                    'the number of processing times.'
                )
            )
        if len(op_starting_dates) != len(proc_times):
            raise ValueError(
                (
                    'The number of operation starting dates must match '
                    'the number of processing times.'
                )
            )
        if len(op_ending_dates) != len(proc_times):
            raise ValueError(
                (
                    'The number of operation ending dates must match '
                    'the number of processing times.'
                )
            )

        ### BASIC INFORMATION ###
        # assign job information
        self.custom_identifier = custom_identifier
        self.job_type: str = 'Job'
        self._dispatcher = dispatcher
        # sum of the proc times of each operation
        # self.total_proc_time: float = sum(proc_times)
        self.total_proc_time: Timedelta = sum(proc_times, Timedelta())

        # inter-process job state parameters
        # first operation scheduled --> released job
        self.is_released: bool = False
        # job's next operation is disposable
        # true for each new job, maybe reworked in future for jobs with
        # a start date later than creation date
        self.is_disposable: bool = True
        self.is_finished: bool = False

        # inter-process time characteristics
        self.time_release = DEFAULT_DATETIME
        self.time_actual_starting = DEFAULT_DATETIME
        self.starting_date_deviation: Timedelta | None = None
        self.time_actual_ending = DEFAULT_DATETIME
        self.ending_date_deviation: Timedelta | None = None
        self.lead_time = Timedelta()
        self.time_creation = DEFAULT_DATETIME

        # current resource location
        self._current_resource: InfrastructureObject | None = None

        # [STATS] Monitoring
        self._stat_monitor = monitors.Monitor(
            env=self._dispatcher.env,
            obj=self,
            init_state=state,
            possible_states=possible_states,
        )

        # register job instance
        current_state = self._stat_monitor.get_current_state()

        env, self._job_id = self._dispatcher.register_job(
            job=self, custom_identifier=self.custom_identifier, state=current_state
        )

        # initialise base class
        name = f'Job (ID: {self._job_id})'
        super().__init__(env=env, name=name, process='', **kwargs)

        ### OPERATIONS ##
        self.operations: deque[Operation] = deque()

        for idx, op_proc_time in enumerate(proc_times):
            op = Operation(
                dispatcher=self._dispatcher,
                job=self,
                proc_time=op_proc_time,
                setup_time=setup_times[idx],
                exec_system_identifier=exec_systems_order[idx],
                target_station_group_identifier=op_target_stations[idx],
                prio=op_prios[idx],
                planned_starting_date=op_starting_dates[idx],
                planned_ending_date=op_ending_dates[idx],
            )
            self.operations.append(op)

        self.open_operations = self.operations.copy()
        self.total_num_ops = len(self.operations)
        self.num_finished_ops: int = 0
        # current and last OP: properties set by method "get_next_operation"
        self.last_op: Operation | None = None
        self.last_proc_time: Timedelta | None = None
        self.last_setup_time: Timedelta | None = None
        self.last_order_time: Timedelta | None = None
        self.current_op: Operation | None = None
        self.current_proc_time: Timedelta | None = None
        self.current_setup_time: Timedelta | None = None
        self.current_order_time: Timedelta | None = None

        # ------- NOT IMPLEMENTED YET -------
        # rank-like property, set if job enters the infrastructure object
        # acts like a counter to allow easy sorting even if queue order is not maintained
        self._obj_entry_idx: int | None = None

        ### ADDITIONAL INFORMATION ###
        self.additional_info = additional_info

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    @property
    def stat_monitor(self) -> monitors.Monitor:
        return self._stat_monitor

    @property
    def job_id(self) -> LoadID:
        return self._job_id

    @property
    def prio(self) -> int | None:
        return self._prio

    @prio.setter
    def prio(
        self,
        new_prio: int,
    ) -> None:
        """setting of priority
        prio can be initialized as None,
        but every change has to be an integer value

        Parameters
        ----------
        new_prio : int
            new priority which should be set

        Raises
        ------
        TypeError
            if other types are provided
        """
        if not isinstance(new_prio, int):
            raise TypeError(
                f'The type of {new_prio} must be >>int<<, but it is {type(new_prio)}.'
            )
        else:
            self._prio = new_prio

    @property
    def obj_entry_idx(self) -> int | None:
        """
        returns the entry index which is set by each infrastructure object
        """
        return self._obj_entry_idx

    @property
    def current_resource(self) -> InfrastructureObject | None:
        """
        returns the current resource on which the job lies
        """
        return self._current_resource

    @current_resource.setter
    def current_resource(self, obj: InfrastructureObject) -> None:
        """setting the current resource object which must be of type InfrastructureObject"""
        if not isinstance(obj, InfrastructureObject):
            raise TypeError(
                f'From {self}: Object >>{obj}<< muste be of type >>InfrastructureObject<<.'
            )
        else:
            self._current_resource = obj
