from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterator
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, Generic, TypeVar, cast
from typing_extensions import override

import numpy as np

from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import SimStatesCommon, SimSystemTypes, TimeUnitsTimedelta
from pyforcesim.errors import SystemNotInitialisedError
from pyforcesim.loggers import loads as logger
from pyforcesim.types import (
    ExpectedArrivalTimes,
    JobGenerationInfo,
    OrderDates,
    OrderTimes,
    SequenceBatchCom,
    StatDistributionInfo,
)

if TYPE_CHECKING:
    from numpy.random._generator import Generator as NPRandomGenerator

    from pyforcesim.simulation import distributions
    from pyforcesim.simulation.environment import (
        InfrastructureObject,
        ProductionArea,
        SimulationEnvironment,
        Source,
        SystemID,
    )
    from pyforcesim.types import (
        DueDate,
        OrderPriority,
        SourceSequence,
    )

A = TypeVar('A', bound='distributions.StatisticalDistribution')
J = TypeVar('J', bound='distributions.StatisticalDistribution')


def calc_due_date(
    starting_date: Datetime,
    total_operational_time: Timedelta,
    factor: float,
) -> DueDate:
    if factor < 0.0:
        raise ValueError('Values smaller than 0 not allowed as factor.')
    time_till_due = factor * total_operational_time
    due_date = starting_date + time_till_due

    return due_date


# ** theory of logistics operating curves
def calc_WIP_relative(
    t: float,
    alpha: float,
) -> float:
    return (1 - (1 - t ** (1 / 4)) ** 4) + alpha * t


def calc_capacity_relative(
    t: float,
) -> float:
    return 1 - (1 - t ** (1 / 4)) ** 4


def calc_t_on_WIP_relative(
    WIP_target: float,
    alpha: float = 10,
    epsilon: float = 1e-3,
) -> float:
    # t between [0;1]
    t: float = 0.5
    WIP_calc = calc_WIP_relative(t, alpha)
    delta = WIP_calc - WIP_target

    t_step: float = 0.1
    step: int = 1
    while abs(delta) > epsilon:
        logger.debug('Delta is: %.4f at step %d', delta, step)
        if delta > 0:
            # decrease t
            t -= t_step / step
            if t < 0:
                t = 0
        else:
            # increase t
            t += t_step / step
            if t > 1:
                t = 1

        WIP_calc = calc_WIP_relative(t, alpha)
        delta = WIP_calc - WIP_target
        step += 1

    logger.debug('Delta before return is: %.4f after step %d', delta, step)

    return t


def calc_t_on_capacity_relative(
    capacity_target: float,
    epsilon: float = 1e-3,
) -> float:
    # t between [0;1]
    t: float = 0.5
    capacity_calc = calc_capacity_relative(t)
    delta = capacity_calc - capacity_target

    t_step: float = 0.1
    step: int = 1
    while abs(delta) > epsilon:
        logger.debug('Delta is: %.4f at step %d', delta, step)
        if delta > 0:
            # decrease t
            t -= t_step / step
            if t < 0:
                t = 0
        else:
            # increase t
            t += t_step / step
            if t > 1:
                t = 1

        capacity_calc = calc_capacity_relative(t)
        delta = capacity_calc - capacity_target
        step += 1

    logger.debug('Delta before return is: %.4f after step %d', delta, step)

    return t


def calc_expected_arrival_time(
    source: InfrastructureObject,
    prod_area: ProductionArea,
    *,
    stat_info_orders: StatDistributionInfo,
    WIP_factor: float,
) -> ExpectedArrivalTimes:
    if not prod_area.initialised:
        raise SystemNotInitialisedError(
            f'System >>{prod_area}<< must be initialised, before WIP related KPIs can be calculated.'
        )

    NORM_TD = pyf_dt.timedelta_from_val(1, TimeUnitsTimedelta.HOURS)
    total_num_proc_stations = prod_area.num_assoc_proc_station
    # statistical information for order times
    mean = stat_info_orders.mean
    std = stat_info_orders.std
    # calc expected value of interval
    # C_S / (C_M/µ + a*(1+(std^2/mean^2))
    prod_area_capa = prod_area.processing_capacities(total=True) / NORM_TD
    source_capa = source.processing_capacity / NORM_TD
    arrival_time_expected_ideal = (source_capa / prod_area_capa) * mean
    n_M = total_num_proc_stations
    arrival_time_expected = source_capa / (
        (prod_area_capa / mean) + n_M * WIP_factor * (1 + std**2 / mean**2)
    )

    return ExpectedArrivalTimes(
        ideal=arrival_time_expected_ideal,
        current=arrival_time_expected,
    )


# order time management
class BaseJobGenerator(Generic[J, A], metaclass=ABCMeta):
    def __init__(
        self,
        env: SimulationEnvironment,
        source: Source,
        dist_order: J,
        dist_arrival: A,
        seed: int | None = None,
    ) -> None:
        """ABC with which load generators can be build
        `retrieve` method must be implemented which yields
        `JobGenerationInfo` objects

        Parameters
        ----------
        env : SimulationEnvironment
            corresponding simulation environment where job generator is used
        seed : int, optional
            seed for random number generator, by default None,
            if None the seed of the associated simulation environment is checked
            and used if available
        """
        # simulation environment
        self._env = env
        self._system_id = self.env.get_system_id()
        # components for random number generation
        if seed is None and self.env.seed is not None:
            seed = self.env.seed
        self._dist_order = dist_order
        self._dist_arrival = dist_arrival

        self._rnd_gen = np.random.default_rng(seed=seed)
        self._seed = seed
        self._source = source
        self._norm_td = pyf_dt.timedelta_from_val(1.0, TimeUnitsTimedelta.HOURS)
        # self._stat_info: StatDistributionInfo | None = None
        # register instance

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} | Env: {self.env.name()} | SystemID: {self.system_id}'
        )

    def __key(self) -> tuple[SystemID, str]:
        return (self.system_id, self.__repr__())

    def __hash__(self) -> int:
        return hash(self.__key())

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def system_id(self) -> SystemID:
        return self._system_id

    @property
    def dist_order(self) -> J:
        return self._dist_order

    @property
    def dist_arrival(self) -> A:
        return self._dist_arrival

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def source(self) -> Source:
        return self._source

    @property
    def rnd_gen(self) -> NPRandomGenerator:
        return self._rnd_gen

    # @property
    # def stat_info(self) -> StatDistributionInfo | None:
    #     return self._stat_info

    # @stat_info.setter
    # def stat_info(
    #     self,
    #     value: StatDistributionInfo,
    # ) -> None:
    #     if not isinstance(value, StatDistributionInfo):
    #         raise TypeError(
    #             (
    #                 f'Type >StatDistributionInfo< must be assigned, '
    #                 f'but value >>{value}<< was of type {type(value)}'
    #             )
    #         )
    #     self._stat_info = value

    @abstractmethod
    def retrieve(self) -> Iterator[SourceSequence]: ...


# TODO: rewrite or remove
# class RandomJobGenerator(BaseJobGenerator):
#     def __init__(
#         self,
#         env: SimulationEnvironment,
#         source: Source,
#         seed: int | None = None,
#         min_proc_time: int = 1,
#         max_proc_time: int = 10,
#         min_setup_time: int = 1,
#         max_setup_time: int = 10,
#         min_prio: OrderPriority = 1,
#         max_prio: OrderPriority = 9,
#     ) -> None:
#         # init base class
#         super().__init__(env=env, source=source, seed=seed)

#         self.min_proc_time = min_proc_time
#         self.max_proc_time = max_proc_time
#         self.min_setup_time = min_setup_time
#         self.max_setup_time = max_setup_time
#         self.min_prio = min_prio
#         self.max_prio = max_prio

#     def gen_rnd_JSSP_inst(
#         self,
#         n_jobs: int,
#         n_machines: int,
#     ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
#         # generate random process time matrix shape=(n_jobs, n_machines)
#         mat_ProcTimes = self.rnd_gen.integers(
#             1, 10, size=(n_jobs, n_machines), dtype=np.uint32
#         )

#         # generate randomly shuffled job machine combinations
#         # machine IDs from 1 to n_machines
#         temp = np.arange(0, n_machines, step=1, dtype=np.uint32)
#         temp = np.expand_dims(temp, axis=0)
#         # repeat dummy line until number n_jobs is reached
#         temp = np.repeat(temp, n_jobs, axis=0)
#         # randomly permute the machine indices job-wise
#         mat_JobMachID = self.rnd_gen.permuted(temp, axis=1)

#         return mat_ProcTimes, mat_JobMachID

#     def retrieve(
#         self,
#         exec_system_ids: Sequence[SystemID],
#         target_station_group_ids: dict[SystemID, Sequence[SystemID]],
#         gen_setup_times: bool = False,
#         time_unit: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
#         gen_prio: bool = False,
#     ) -> Iterator[JobGenerationInfo]:
#         """Generic function to generate processing times and execution flow of a job object"""

#         n_objects = len(exec_system_ids)

#         while True:
#             # ** execution order
#             # randomly permute the execution systems indices
#             execution_systems = cast(
#                 list[SystemID], self.rnd_gen.permuted(exec_system_ids).tolist()
#             )

#             station_groups: list[SystemID] | None = None
#             station_groups = []
#             for exec_system_id in execution_systems:
#                 # multiple candidates: random choice
#                 candidates = target_station_group_ids[exec_system_id]

#                 if len(candidates) > 1:
#                     candidate = cast(SystemID, self.rnd_gen.choice(candidates))
#                 else:
#                     candidate = candidates[0]

#                 station_groups.append(candidate)

#             # ** order times
#             # processing times
#             proc_times: list[Timedelta] = []
#             proc_times_time_unit = cast(
#                 list[int],
#                 self.rnd_gen.integers(
#                     self.min_proc_time, self.max_proc_time, size=n_objects, dtype=np.uint32
#                 ).tolist(),
#             )
#             for time in proc_times_time_unit:
#                 td = pyf_dt.timedelta_from_val(val=time, time_unit=time_unit)
#                 proc_times.append(td)

#             # setup times
#             setup_times: list[Timedelta]
#             if gen_setup_times:
#                 setup_times = []
#                 setup_times_time_unit = cast(
#                     list[int],
#                     self.rnd_gen.integers(
#                         self.min_setup_time,
#                         self.max_setup_time,
#                         size=n_objects,
#                         dtype=np.uint32,
#                     ).tolist(),
#                 )
#                 for time in setup_times_time_unit:
#                     td = pyf_dt.timedelta_from_val(val=time, time_unit=time_unit)
#                     setup_times.append(td)
#             else:
#                 setup_times = [Timedelta()] * n_objects

#             prio: OrderPriority | None = None
#             if gen_prio:
#                 prio = self.gen_prio()

#             job_gen_info = JobGenerationInfo(
#                 custom_id=None,
#                 execution_systems=execution_systems,
#                 station_groups=station_groups,
#                 order_time=OrderTimes(proc=proc_times, setup=setup_times),
#                 dates=OrderDates(),
#                 prio=prio,
#                 current_state=SimStatesCommon.INIT,
#             )

#             yield job_gen_info

#     def gen_prio(
#         self,
#     ) -> OrderPriority:
#         """Generates a single priority score

#         Parameters
#         ----------
#         lowest_prio : int
#             lowest available priority
#         highest_prio : int
#             highest available priority

#         Returns
#         -------
#         int
#             randomly chosen priority between lowest and highest value
#         """
#         return int(self.rnd_gen.integers(low=self.min_prio, high=self.max_prio))


# ** special sequence generation
class ProductionSequence(
    BaseJobGenerator['distributions.Uniform', 'distributions.Exponential']
):
    def __init__(
        self,
        env: SimulationEnvironment,
        dist_order: distributions.Uniform,
        dist_arrival: distributions.Exponential,
        source: Source,
        seed: int | None = None,
    ) -> None:
        # init base class
        super().__init__(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            seed=seed,
        )

    def _get_order_times(
        self,
        time_operational: Timedelta,
        due_date_factor: float = 1.0,
    ) -> OrderDates:
        # planned starting time = current time
        # planned ending = calculated due date
        curr_time = self.env.t_as_dt()
        due_date = calc_due_date(
            starting_date=curr_time,
            total_operational_time=time_operational,
            factor=due_date_factor,
        )
        return OrderDates(
            starting_planned=[curr_time],
            ending_planned=[due_date],
        )


class SequenceSinglePA(ProductionSequence):
    def __init__(
        self,
        env: SimulationEnvironment,
        dist_order: distributions.Uniform,
        dist_arrival: distributions.Exponential,
        source: Source,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            seed=seed,
        )
        # associated production area
        self._prod_area_id = prod_area_id
        self._prod_area = self.env.infstruct_mgr.get_system_by_id(
            system_type=SimSystemTypes.PRODUCTION_AREA,
            system_id=self._prod_area_id,
        )
        self.source.register_job_generator(self)

    def __repr__(self) -> str:
        return super().__repr__() + f' | ProductionAreaID: {self._prod_area_id}'

    @property
    def prod_area_id(self) -> SystemID:
        return self._prod_area_id

    @property
    def prod_area(self) -> ProductionArea:
        return self._prod_area

    def recalculate_expected_arrival_time(
        self,
        WIP_relative: float,
    ) -> None:
        arrival_times = calc_expected_arrival_time(
            source=self.source,
            prod_area=self.prod_area,
            stat_info_orders=self.dist_order.stat_info,
            WIP_factor=WIP_relative,
        )
        arrival_time_expected = arrival_times.current
        # update distribution
        StatParams = self.dist_arrival.get_param_type()
        new_params = StatParams(scale=arrival_time_expected)
        self.dist_arrival.set_params(new_params)

    def recalculate_lead_time_delta(
        self,
        WIP_relative: float,
    ) -> None:
        self.prod_area.set_lead_time_current(
            WIP_relative=WIP_relative,
            order_time_stats_info=self.dist_order.stat_info,
            alpha=self.prod_area.WIP_alpha,
            recalculate_delta=True,
        )


class ConstantSequenceSinglePA(SequenceSinglePA):
    def __init__(
        self,
        env: SimulationEnvironment,
        dist_order: distributions.Uniform,
        dist_arrival: distributions.Exponential,
        source: Source,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            prod_area_id=prod_area_id,
            seed=seed,
        )

    @override
    def retrieve(
        self,
        due_date_factor: float = 1.0,
    ) -> Iterator[SourceSequence]:
        """Generates a constant sequence of job generation infos

        Parameters
        ----------
        target_obj : Source
            associated source object

        Yields
        ------
        Iterator[SourceSequence]
            job generation infos, processing time
        """
        # request StationGroupIDs by ProdAreaID in StationGroup database
        stat_groups = self.prod_area.subsystems_as_tuple()

        logger.debug('stat_groups: %s', stat_groups)

        # number of all processing stations in associated production area
        total_num_proc_stations = self._prod_area.num_assoc_proc_station

        logger.debug('total_num_proc_stations: %s', total_num_proc_stations)

        # order time equally distributed between all station within given ProductionArea
        # source distributes loads in round robin principle
        # order time for each station has to be the order time of the source
        # times the number of stations the source delivers to
        order_time_source = self.source.order_time
        overall_time = order_time_source * total_num_proc_stations

        logger.debug('station_order_time: %s', overall_time)

        # generate endless sequence
        while True:
            # iterate over all StationGroups
            for stat_group in stat_groups:
                # generate job for each ProcessingStation in StationGroup
                for _ in range(stat_group.num_assoc_proc_station):
                    # generate random distribution of setup and processing time
                    setup_time_percentage = self.dist_order.sample()
                    setup_time = setup_time_percentage * overall_time
                    # round to next full minute
                    setup_time = pyf_dt.round_td_by_seconds(
                        td=setup_time, round_to_next_seconds=60
                    )
                    proc_time = overall_time - setup_time
                    order_dates = self._get_order_times(
                        time_operational=overall_time,
                        due_date_factor=due_date_factor,
                    )
                    stat_group_id = stat_group.system_id
                    job_gen_info = JobGenerationInfo(
                        custom_id=None,
                        execution_systems=[self._prod_area_id],
                        station_groups=[stat_group_id],
                        order_time=OrderTimes(proc=[proc_time], setup=[setup_time]),
                        dates=order_dates,
                        prio=None,
                        current_state=SimStatesCommon.INIT,
                    )
                    source_proc_time = self.source.obtain_order_time()

                    yield job_gen_info, source_proc_time


class VariableSequenceSinglePA(SequenceSinglePA):
    def __init__(
        self,
        env: SimulationEnvironment,
        dist_order: distributions.Uniform,
        dist_arrival: distributions.Exponential,
        source: Source,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            prod_area_id=prod_area_id,
            seed=seed,
        )

    @override
    def retrieve(
        self,
        target_obj: Source,
        delta_percentage: float,
        delta_lower_bound: float = 0.1,
        delta_upper_bound: float = 0.5,
        due_date_factor: float = 1.0,
    ) -> Iterator[SourceSequence]:
        """Generates a variable sequence of job generation infos,
        evenly distributed within a given percentage range where the middle
        of the range is the order time of a ideal sequence for the production
        area

        Parameters
        ----------
        target_obj : Source
            associated source object
        delta_percentage : float
            percentage delta measured by the distance from value 1.0
            (ideal sequence), mus lie in [delta_lower_bound, delta_upper_bound]
        delta_lower_bound : float, optional
            allowed lower bound for delta percentage, by default 0.1
        delta_upper_bound : float, optional
            allowed upper bound for delta percentage, by default 0.5

        Yields
        ------
        Iterator[SourceSequence]
            job generation infos, processing time

        Raises
        ------
        ValueError
            if delta percentage exceeds provided bounds
        """

        # value bounds
        if delta_percentage < delta_lower_bound or delta_percentage > delta_upper_bound:
            raise ValueError(
                (
                    f'Percentage delta must lie between '
                    f'{delta_lower_bound} and {delta_upper_bound}'
                )
            )
        lower_percentage = 1.0 - delta_percentage
        upper_percentage = 1.0 + delta_percentage
        # request StationGroupIDs by ProdAreaID in StationGroup database
        stat_groups = self.prod_area.subsystems_as_tuple()
        logger.debug('stat_groups: %s', stat_groups)

        # number of all processing stations in associated production area
        total_num_proc_stations = self._prod_area.num_assoc_proc_station

        logger.debug('total_num_proc_stations: %s', total_num_proc_stations)

        # order time equally distributed between all station within given ProductionArea
        # source distributes loads in round robin principle
        # order time for each station has to be the order time of the source
        # times the number of stations the source delivers to
        order_time_source = target_obj.order_time
        overall_time_ideal = order_time_source * total_num_proc_stations

        logger.debug('station_order_time: %s', overall_time_ideal)

        # generate endless sequence
        while True:
            # iterate over all StationGroups
            for stat_group in stat_groups:
                # generate job for each ProcessingStation in StationGroup
                for _ in range(stat_group.num_assoc_proc_station):
                    # generate random difference in total order time
                    order_time_percentage = self.rnd_gen.uniform(
                        low=lower_percentage,
                        high=upper_percentage,
                    )
                    overall_time = order_time_percentage * overall_time_ideal
                    overall_time = pyf_dt.round_td_by_seconds(
                        td=overall_time, round_to_next_seconds=60
                    )
                    # generate random distribution of setup and processing time
                    setup_time_percentage = self.rnd_gen.uniform(low=0.1, high=0.8)
                    setup_time = setup_time_percentage * overall_time
                    # round to next full minute
                    setup_time = pyf_dt.round_td_by_seconds(
                        td=setup_time, round_to_next_seconds=60
                    )
                    proc_time = overall_time - setup_time
                    order_dates = self._get_order_times(
                        time_operational=overall_time,
                        due_date_factor=due_date_factor,
                    )
                    stat_group_id = stat_group.system_id
                    job_gen_info = JobGenerationInfo(
                        custom_id=None,
                        execution_systems=[self._prod_area_id],
                        station_groups=[stat_group_id],
                        order_time=OrderTimes(proc=[proc_time], setup=[setup_time]),
                        dates=order_dates,
                        prio=None,
                        current_state=SimStatesCommon.INIT,
                    )
                    source_proc_time = target_obj.obtain_order_time()

                    yield job_gen_info, source_proc_time


class WIPSequenceSinglePA(SequenceSinglePA):
    def __init__(
        self,
        env: SimulationEnvironment,
        dist_order: distributions.Uniform,
        dist_arrival: distributions.Exponential,
        source: Source,
        prod_area_id: SystemID,
        seed: int | None = None,
        *,
        WIP_relative_planned: float,
        WIP_alpha: float = 10,
    ) -> None:
        super().__init__(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            prod_area_id=prod_area_id,
            seed=seed,
        )
        self.WIP_relative_planned = WIP_relative_planned
        self.WIP_alpha = WIP_alpha

    @override
    def retrieve(
        self,
        WIP_relative: float = 0,
        job_pool_size: int = 1,
        shuffle_job_pool: bool = True,
        random_due_date_diff: bool = False,
    ) -> Generator[SourceSequence, SequenceBatchCom, None]:
        """(-1) < factor_WIP < 0: underload condition, factor_WIP <= (-1) only
        allowed, if WIP would initially be higher than ideal
        factor_WIP = 0: ideal condition
        factor_WIP > 0: overload condition

        Parameters
        ----------
        target_obj : Source
            _description_
        uniform_lb : float
            _description_
        uniform_ub : float
            _description_
        factor_WIP : float, optional
            _description_, by default 0

        Yields
        ------
        Iterator[SourceSequence]
            _description_
        """
        self.recalculate_expected_arrival_time(WIP_relative)
        # # number of all processing stations in associated production area
        # total_num_proc_stations = self.prod_area.num_assoc_proc_station
        # # statistical information
        # mean = self.stat_info.mean
        # std = self.stat_info.std
        # # calc expected value of interval
        # # C_S / (C_M/µ + a*(1+(std^2/mean^2))
        # prod_area_capa = self.prod_area.processing_capacities(total=True) / self._norm_td
        # source_capa = target_obj.processing_capacity / self._norm_td
        # exp_val_interval_ideal = (source_capa / prod_area_capa) * mean
        # n_M = total_num_proc_stations
        # exp_val_interval = source_capa / (
        #     (prod_area_capa / mean) + n_M * factor_WIP * (1 + std**2 / mean**2)
        # )
        # determine condition
        # underload condition: sustain longer time spans to hold it
        # otherwise ideal condition is met again
        # overload_condition: bool = False
        # if factor_WIP > 0:
        #     # later used to switch back to ideal sequence
        #     overload_condition = True

        # request StationGroupIDs by ProdAreaID in StationGroup database
        stat_groups = self.prod_area.subsystems_as_tuple()
        logger.debug('stat_groups: %s', stat_groups)
        logger.debug('total_num_proc_stations: %s', self.prod_area.num_assoc_proc_station)

        # duration for WIP build-up/tear-down phase
        # implicit in equation: 1 day
        # TODO changed for test of WIP regulation
        # duration_non_ideal_sequence = pyf_dt.timedelta_from_val(104, TimeUnitsTimedelta.WEEKS)
        # td_zero = Timedelta()

        # ** planned lead time: identical for each job
        self.prod_area.set_lead_time_planned(
            WIP_relative=self.WIP_relative_planned,
            order_time_stats_info=self.dist_order.stat_info,
            alpha=self.WIP_alpha,
        )
        lead_time_planned = self.prod_area.lead_time_planned
        logger.info(
            '[LOADS] ProdArea: %s, Calculated planned lead time: %s',
            self.prod_area,
            lead_time_planned,
        )
        self.recalculate_lead_time_delta(WIP_relative)
        logger.debug(
            '[LOADS] ProdArea: %s, Calculated lead time delta: %s',
            self.prod_area,
            self.prod_area.lead_time_delta,
        )
        # random change in planned due date
        upper_bound_dev = np.sqrt(3)
        lower_bound_dev = (-1) * upper_bound_dev

        # batching generator
        gen_com = SequenceBatchCom()
        batching_seq = generate_job_pool(pool_size=job_pool_size)

        # generate endless sequence
        while True:
            # iterate over all StationGroups
            for stat_group in stat_groups:
                # generate job for each ProcessingStation in StationGroup
                for _ in range(stat_group.num_assoc_proc_station):
                    # generate random order time from stat distribution
                    overall_time_float = self.dist_order.sample()
                    overall_time = pyf_dt.timedelta_from_val(
                        overall_time_float, TimeUnitsTimedelta.HOURS
                    )
                    overall_time = pyf_dt.round_td_by_seconds(
                        td=overall_time, round_to_next_seconds=60
                    )
                    # generate constant/random distribution of setup and processing time
                    # setup_time_percentage = self.rnd_gen.uniform(low=0.1, high=0.8)
                    # constant setup time percentage
                    setup_time_percentage = 0.2
                    setup_time = setup_time_percentage * overall_time
                    setup_time = pyf_dt.round_td_by_seconds(
                        td=setup_time, round_to_next_seconds=60
                    )
                    proc_time = overall_time - setup_time

                    # ideal situation is assumed: therefore ideal WIP is ensured which
                    # leads to utilisation of nearly 100 percent (input equals output)
                    # applies in this case: mean lead time = mean order time
                    # three scenarios:
                    # (1) WIP = WIP_ideal: actual lead time should equal
                    # theoretical planned value
                    # (2) WIP < WIP_ideal: actual lead time should also be equal to ideal
                    # (3) WIP > WIP_ideal: actual lead time should be greater than
                    # theoretical planned value
                    # due_date_factor: float = 1.0

                    # ** planned dates
                    # calc based on planned values: set relative WIP target to 1.5
                    curr_time = self.env.t_as_dt()

                    # source processing time
                    interval_td = self.dist_arrival.sample_timedelta(round_to_minutes=True)

                    # send current time and interval to generator
                    # retrieve adapted current time
                    gen_com.start_date = curr_time
                    gen_com.interval = interval_td
                    next(batching_seq)
                    com = batching_seq.send(gen_com)  # type: ignore
                    assert com is not None, 'response of job pool generator is None'

                    assert com.adapted_date is not None, 'adapted date is None'
                    adapted_time = com.adapted_date
                    due_date_planned = adapted_time + lead_time_planned

                    # random change in planned due date
                    if random_due_date_diff:
                        hours_deviation = self.rnd_gen.uniform(
                            lower_bound_dev, upper_bound_dev
                        )
                        planned_ending_dev = pyf_dt.timedelta_from_val(
                            hours_deviation, time_unit=TimeUnitsTimedelta.HOURS
                        )
                        due_date_planned += planned_ending_dev

                    order_dates = OrderDates(
                        starting_planned=[adapted_time],
                        ending_planned=[due_date_planned],
                    )

                    job_gen_info = JobGenerationInfo(
                        custom_id=None,
                        execution_systems=[self._prod_area_id],
                        station_groups=[stat_group.system_id],
                        order_time=OrderTimes(proc=[proc_time], setup=[setup_time]),
                        dates=order_dates,
                        prio=None,
                        current_state=SimStatesCommon.INIT,
                    )
                    logger.debug('[LOADS] Generated new job at %s', curr_time)

                    # send job_gen_info; interval_td already known
                    com.job_gen_info = job_gen_info
                    # next(batching_seq)
                    com = batching_seq.send(gen_com)  # type: ignore

                    # retrieve job pool if fully filled else not available
                    assert com is not None, 'response of batch generator is None'
                    batch = com.batch

                    if batch is not None:
                        if shuffle_job_pool and len(batch) > 1:
                            self.rnd_gen.shuffle(batch)  # type: ignore
                        yield from batch

                    # assert (
                    #     self.arrival_time_expected is not None
                    # ), 'Expected value for interval not set'
                    # interval_hours = self.rnd_gen.exponential(
                    #     scale=self.arrival_time_expected
                    # )  # in hours
                    # # transform to TD
                    # interval_td = pyf_dt.timedelta_from_val(
                    #     interval_hours, TimeUnitsTimedelta.HOURS
                    # )
                    # interval_td = pyf_dt.round_td_by_seconds(
                    #     interval_td,
                    #     round_to_next_seconds=60,
                    # )

                    # if overload_condition and duration_non_ideal_sequence > td_zero:
                    #     duration_non_ideal_sequence -= interval_td
                    # elif overload_condition and duration_non_ideal_sequence <= td_zero:
                    #     exp_val_interval = exp_val_interval_ideal

                    # yield job_gen_info, interval_td


def generate_job_pool(
    pool_size: int = 1,
) -> Generator[SequenceBatchCom | None, None, None]:
    batch: list[SourceSequence] = []
    adapted_start_date: Datetime | None = None

    while True:  # generate endlessly
        # receive starting date and interval
        com = yield
        com = cast(SequenceBatchCom, com)
        if adapted_start_date is None:
            assert com.start_date is not None, 'starting date None'
            adapted_start_date = com.start_date
        com.adapted_date = adapted_start_date
        assert com.interval is not None, 'interval None'
        curr_interval = com.interval
        adapted_start_date = adapted_start_date + curr_interval
        # send adapted date, receive job generation info
        com = yield com
        com = cast(SequenceBatchCom, com)
        assert com.job_gen_info is not None, 'JobGenInfo None'
        batch.append((com.job_gen_info, curr_interval))

        if len(batch) == pool_size:
            com.batch = batch
            yield com
            batch = []
            adapted_start_date = None
        else:
            com.batch = None
            yield com
