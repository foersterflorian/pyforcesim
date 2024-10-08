from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np
import numpy.typing as npt
from numpy.random._generator import Generator as NPRandomGenerator

from pyforcesim import datetime as pyf_dt
from pyforcesim import loggers
from pyforcesim.constants import SimStatesCommon, SimSystemTypes, TimeUnitsTimedelta
from pyforcesim.simulation.environment import SimulationEnvironment, SystemID
from pyforcesim.types import (
    DueDate,
    JobGenerationInfo,
    OrderDates,
    OrderPriority,
    OrderTimes,
)

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        ProductionArea,
        SimulationEnvironment,
        Source,
        SystemID,
    )


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


# order time management
class BaseJobGenerator(ABC):
    def __init__(
        self,
        env: SimulationEnvironment,
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
        # components for random number generation
        if seed is None and self.env.seed is not None:
            seed = self.env.seed
        self._rnd_gen = np.random.default_rng(seed=seed)
        self._seed = seed

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} | Env: {self.env.name()} | Seed: {self.seed}'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def rnd_gen(self) -> NPRandomGenerator:
        return self._rnd_gen

    @abstractmethod
    def retrieve(self) -> Iterator[JobGenerationInfo]: ...


# TODO: cleanup and remove outdated methods
class RandomJobGenerator(BaseJobGenerator):
    def __init__(
        self,
        env: SimulationEnvironment,
        seed: int | None = None,
        min_proc_time: int = 1,
        max_proc_time: int = 10,
        min_setup_time: int = 1,
        max_setup_time: int = 10,
        min_prio: OrderPriority = 1,
        max_prio: OrderPriority = 9,
    ) -> None:
        # init base class
        super().__init__(env=env, seed=seed)

        self.min_proc_time = min_proc_time
        self.max_proc_time = max_proc_time
        self.min_setup_time = min_setup_time
        self.max_setup_time = max_setup_time
        self.min_prio = min_prio
        self.max_prio = max_prio

    def gen_rnd_JSSP_inst(
        self,
        n_jobs: int,
        n_machines: int,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        # generate random process time matrix shape=(n_jobs, n_machines)
        mat_ProcTimes = self.rnd_gen.integers(
            1, 10, size=(n_jobs, n_machines), dtype=np.uint32
        )

        # generate randomly shuffled job machine combinations
        # machine IDs from 1 to n_machines
        temp = np.arange(0, n_machines, step=1, dtype=np.uint32)
        temp = np.expand_dims(temp, axis=0)
        # repeat dummy line until number n_jobs is reached
        temp = np.repeat(temp, n_jobs, axis=0)
        # randomly permute the machine indices job-wise
        mat_JobMachID = self.rnd_gen.permuted(temp, axis=1)

        return mat_ProcTimes, mat_JobMachID

    def retrieve(
        self,
        exec_system_ids: Sequence[SystemID],
        target_station_group_ids: dict[SystemID, Sequence[SystemID]],
        gen_setup_times: bool = False,
        time_unit: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
        gen_prio: bool = False,
    ) -> Iterator[JobGenerationInfo]:
        """Generic function to generate processing times and execution flow of a job object"""

        n_objects = len(exec_system_ids)

        while True:
            # ** execution order
            # randomly permute the execution systems indices
            execution_systems = cast(
                list[SystemID], self.rnd_gen.permuted(exec_system_ids).tolist()
            )

            station_groups: list[SystemID] | None = None
            station_groups = []
            for exec_system_id in execution_systems:
                # multiple candidates: random choice
                candidates = target_station_group_ids[exec_system_id]

                if len(candidates) > 1:
                    candidate = cast(SystemID, self.rnd_gen.choice(candidates))
                else:
                    candidate = candidates[0]

                station_groups.append(candidate)

            # ** order times
            # processing times
            proc_times: list[Timedelta] = []
            proc_times_time_unit = cast(
                list[int],
                self.rnd_gen.integers(
                    self.min_proc_time, self.max_proc_time, size=n_objects, dtype=np.uint32
                ).tolist(),
            )
            for time in proc_times_time_unit:
                td = pyf_dt.timedelta_from_val(val=time, time_unit=time_unit)
                proc_times.append(td)

            # setup times
            setup_times: list[Timedelta]
            if gen_setup_times:
                setup_times = []
                setup_times_time_unit = cast(
                    list[int],
                    self.rnd_gen.integers(
                        self.min_setup_time,
                        self.max_setup_time,
                        size=n_objects,
                        dtype=np.uint32,
                    ).tolist(),
                )
                for time in setup_times_time_unit:
                    td = pyf_dt.timedelta_from_val(val=time, time_unit=time_unit)
                    setup_times.append(td)
            else:
                setup_times = [Timedelta()] * n_objects

            prio: OrderPriority | None = None
            if gen_prio:
                prio = self.gen_prio()

            job_gen_info = JobGenerationInfo(
                custom_id=None,
                execution_systems=execution_systems,
                station_groups=station_groups,
                order_time=OrderTimes(proc=proc_times, setup=setup_times),
                dates=OrderDates(),
                prio=prio,
                current_state=SimStatesCommon.INIT,
            )

            yield job_gen_info

    def gen_prio(
        self,
    ) -> OrderPriority:
        """Generates a single priority score

        Parameters
        ----------
        lowest_prio : int
            lowest available priority
        highest_prio : int
            highest available priority

        Returns
        -------
        int
            randomly chosen priority between lowest and highest value
        """
        return int(self.rnd_gen.integers(low=self.min_prio, high=self.max_prio))


# ** special sequence generation
class ProductionSequence(BaseJobGenerator):
    def __init__(
        self,
        env: SimulationEnvironment,
        seed: int | None = None,
    ) -> None:
        # init base class
        super().__init__(env=env, seed=seed)


class SequenceSinglePA(ProductionSequence):
    def __init__(
        self,
        env: SimulationEnvironment,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(env=env, seed=seed)

        # associated production area
        self._prod_area_id = prod_area_id
        self._prod_area = self.env.infstruct_mgr.get_system_by_id(
            system_type=SimSystemTypes.PRODUCTION_AREA,
            system_id=self._prod_area_id,
        )

    def __repr__(self) -> str:
        return super().__repr__() + f' | ProductionAreaID: {self._prod_area_id}'

    @property
    def prod_area_id(self) -> SystemID:
        return self._prod_area_id

    @property
    def prod_area(self) -> ProductionArea:
        return self._prod_area


class ConstantSequenceSinglePA(SequenceSinglePA):
    def __init__(
        self,
        env: SimulationEnvironment,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(env=env, prod_area_id=prod_area_id, seed=seed)

    @override
    def retrieve(
        self,
        target_obj: Source,
    ) -> Iterator[JobGenerationInfo]:
        """Generates a constant sequence of job generation infos

        Parameters
        ----------
        target_obj : Source
            associated source object

        Yields
        ------
        Iterator[JobGenerationInfo]
            job generation infos
        """
        # request StationGroupIDs by ProdAreaID in StationGroup database
        stat_groups = self.prod_area.subsystems_as_tuple()

        loggers.loads.debug('stat_groups: %s', stat_groups)

        # number of all processing stations in associated production area
        total_num_proc_stations = self._prod_area.num_assoc_proc_station

        loggers.loads.debug('total_num_proc_stations: %s', total_num_proc_stations)

        # order time equally distributed between all station within given ProductionArea
        # source distributes loads in round robin principle
        # order time for each station has to be the order time of the source
        # times the number of stations the source delivers to
        order_time_source = target_obj.order_time
        overall_time = order_time_source * total_num_proc_stations

        loggers.loads.debug('station_order_time: %s', overall_time)

        # generate endless sequence
        while True:
            # iterate over all StationGroups
            for stat_group in stat_groups:
                # generate job for each ProcessingStation in StationGroup
                for _ in range(stat_group.num_assoc_proc_station):
                    # generate random distribution of setup and processing time
                    setup_time_percentage = self.rnd_gen.uniform(low=0.1, high=0.8)
                    setup_time = setup_time_percentage * overall_time
                    # round to next full minute
                    setup_time = pyf_dt.round_td_by_seconds(
                        td=setup_time, round_to_next_seconds=60
                    )
                    proc_time = overall_time - setup_time
                    # StationGroupID
                    stat_group_id = stat_group.system_id
                    job_gen_info = JobGenerationInfo(
                        custom_id=None,
                        execution_systems=[self._prod_area_id],
                        station_groups=[stat_group_id],
                        order_time=OrderTimes(proc=[proc_time], setup=[setup_time]),
                        dates=OrderDates(),
                        prio=None,
                        current_state=SimStatesCommon.INIT,
                    )
                    yield job_gen_info


class VariableSequenceSinglePA(SequenceSinglePA):
    def __init__(
        self,
        env: SimulationEnvironment,
        prod_area_id: SystemID,
        seed: int | None = None,
    ) -> None:
        super().__init__(env=env, prod_area_id=prod_area_id, seed=seed)

    @override
    def retrieve(
        self,
        target_obj: Source,
        delta_percentage: float,
        delta_lower_bound: float = 0.1,
        delta_upper_bound: float = 0.5,
        due_date_factor: float = 1.0,
    ) -> Iterator[JobGenerationInfo]:
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
        Iterator[JobGenerationInfo]
            job generation infos

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
        loggers.loads.debug('stat_groups: %s', stat_groups)

        # number of all processing stations in associated production area
        total_num_proc_stations = self._prod_area.num_assoc_proc_station

        loggers.loads.debug('total_num_proc_stations: %s', total_num_proc_stations)

        # order time equally distributed between all station within given ProductionArea
        # source distributes loads in round robin principle
        # order time for each station has to be the order time of the source
        # times the number of stations the source delivers to
        order_time_source = target_obj.order_time
        overall_time_ideal = order_time_source * total_num_proc_stations

        loggers.loads.debug('station_order_time: %s', overall_time_ideal)

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
                    # TODO add OrderDates
                    # planned starting time = current time
                    # planned ending = calculated due date
                    curr_time = self.env.t_as_dt()
                    due_date = calc_due_date(
                        starting_date=curr_time,
                        total_operational_time=overall_time,
                        factor=due_date_factor,
                    )
                    order_dates = OrderDates(
                        starting_planned=[curr_time],
                        ending_planned=[due_date],
                    )
                    # StationGroupID
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
                    yield job_gen_info
