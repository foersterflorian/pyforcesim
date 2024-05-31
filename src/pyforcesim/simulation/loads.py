from __future__ import annotations

import random
import typing
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta as Timedelta
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy.random._generator import Generator as NPRandomGenerator

from pyforcesim import loggers
from pyforcesim.constants import SimSystemTypes, TimeUnitsTimedelta
from pyforcesim.datetime import DTManager

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        CustomID,
        ProductionArea,
        SimulationEnvironment,
        StationGroup,
        SystemID,
    )


# order time management
@dataclass
class OrderTime:
    """Dataclass to manage order time components"""

    proc: Timedelta
    setup: Timedelta


class BaseGenerator:
    def __init__(
        self,
        env: SimulationEnvironment,
        seed: int = 42,
    ) -> None:
        """
        seed: seed value for random number generator
        """
        # simulation environment
        self._env = env
        # components for random number generation
        self._np_rnd_gen: NPRandomGenerator = np.random.default_rng(seed=seed)
        self._seed = seed
        random.seed(self._seed)
        # advanced date handling
        self._dt_mgr: DTManager = DTManager()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} | Env: {self._env.name()} | Seed: {self._seed}'

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def seed(self) -> int:
        return self._seed


# TODO: cleanup and remove outdated methods
class RandomJobGenerator(BaseGenerator):
    def __init__(
        self,
        env: SimulationEnvironment,
        seed: int = 42,
        **kwargs,
    ) -> None:
        # init base class
        super().__init__(env=env, seed=seed, **kwargs)

    def gen_rnd_JSSP_inst(
        self,
        n_jobs: int,
        n_machines: int,
    ) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
        """
        Generates random job shop instance with given number of and machines
        - each job on all machines
        - max processing time = 9

        Output:
        n_jobs: number of jobs
        n_machines: number of machines
        n_tasks: number of tasks
        mat_ProcTimes: matrix of processing times | shape=(n_jobs,n_machines)
        mat_JobMachID: matrix of machine IDs per job starting
            by index 1 | shape=(n_jobs,n_machines)
        mat_OpID: matrix of operation IDs starting
            by index 1 | shape=(n_jobs,n_machines)
        """
        # generate random process time matrix shape=(n_jobs, n_machines)
        mat_ProcTimes = self._np_rnd_gen.integers(
            1, 10, size=(n_jobs, n_machines), dtype=np.uint16
        )

        # generate randomly shuffled job machine combinations
        # machine IDs from 1 to n_machines
        temp = np.arange(0, n_machines, step=1, dtype=np.uint16)
        temp = np.expand_dims(temp, axis=0)
        # repeat dummy line until number n_jobs is reached
        temp = np.repeat(temp, n_jobs, axis=0)
        # randomly permute the machine indices job-wise
        mat_JobMachID = self._np_rnd_gen.permuted(temp, axis=1)

        # generate operation ID matrix
        # not mandatory because operations are registered in the environment's dispatcher
        # n_ops = n_jobs * n_machines
        # temp2 = np.arange(0, (n_ops), step=1, dtype=np.uint16)
        # mat_OpID = temp2.reshape(n_jobs, -1)

        return mat_ProcTimes, mat_JobMachID

    def gen_rnd_job(
        self,
        n_machines: int,
    ) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
        """generates random job with machine IDs
        [OUTDATED] Should be replaced by the more generic 'gen_rnd_job_by_ids' method
        which uses any IDs provided as NumPy array

        Parameters
        ----------
        n_machines : int
            _description_

        Returns
        -------
        tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]
            _description_
        """

        # generate random process time matrix shape=(n_machines)
        mat_ProcTimes = self._np_rnd_gen.integers(1, 10, size=n_machines, dtype=np.uint16)

        # generate randomly shuffled job machine combinations
        # machine IDs from 1 to n_machines
        temp = np.arange(0, n_machines, step=1, dtype=np.uint16)
        # randomly permute the machine indices job-wise
        mat_JobMachID = self._np_rnd_gen.permuted(temp)

        return mat_ProcTimes, mat_JobMachID

    def gen_rnd_job_by_ids(
        self,
        exec_system_ids: Sequence[CustomID],
        target_station_group_ids: dict[CustomID, Sequence[CustomID]] | None = None,
        min_proc_time: int = 1,
        max_proc_time: int = 10,
        gen_setup_times: bool = False,
        min_setup_time: int = 1,
        max_setup_time: int = 10,
        time_unit: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
    ) -> tuple[
        list[CustomID], list[CustomID] | None, list[Timedelta], list[Timedelta] | None
    ]:
        """Generic function to generate processing times and execution flow of a job object"""
        n_objects = len(exec_system_ids)

        # processing times
        proc_times: list[Timedelta] = list()
        proc_times_time_unit: list[int] = self._np_rnd_gen.integers(
            min_proc_time, max_proc_time, size=n_objects, dtype=np.uint16
        ).tolist()
        for time in proc_times_time_unit:
            # build timedelta object
            td = self._dt_mgr.timedelta_from_val(val=time, time_unit=time_unit)
            proc_times.append(td)

        # setup times
        setup_times: list[Timedelta] = list()
        setup_times_time_unit: list[int] | None = None
        if gen_setup_times:
            setup_times_time_unit = typing.cast(
                list[int],
                self._np_rnd_gen.integers(
                    min_setup_time, max_setup_time, size=n_objects, dtype=np.uint16
                ).tolist(),
            )
            for time in setup_times_time_unit:
                # build timedelta object
                td = self._dt_mgr.timedelta_from_val(val=time, time_unit=time_unit)
                # append object
                setup_times.append(td)

        # randomly permute the execution systems indices
        job_ex_order = self._np_rnd_gen.permuted(exec_system_ids).tolist()

        job_target_station_groups: list[CustomID] | None = None
        if target_station_group_ids is not None:
            job_target_station_groups = list()

            for exec_system_id in job_ex_order:
                # multiple candidates: random choice
                candidates = target_station_group_ids[exec_system_id]

                if len(candidates) > 1:
                    # candidate = self._np_rnd_gen.choice(candidates)
                    candidate = random.choice(candidates)
                # only one entry
                else:
                    candidate = candidates[0]

                job_target_station_groups.append(candidate)

        return job_ex_order, job_target_station_groups, proc_times, setup_times

    def gen_prio(
        self,
        lowest: int = 1,
        highest: int = 9,
    ) -> int:
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
        return int(self._np_rnd_gen.integers(low=lowest, high=highest))


# ** sequence generation


class ProductionSequence(BaseGenerator):
    def __init__(
        self,
        env: SimulationEnvironment,
        seed: int = 42,
        **kwargs,
    ) -> None:
        # init base class
        super().__init__(env=env, seed=seed, **kwargs)


class ProductionSequenceSinglePA(ProductionSequence):
    def __init__(
        self,
        env: SimulationEnvironment,
        prod_area_id: SystemID,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(env=env, seed=seed, **kwargs)

        # associated production area
        self._prod_area_id = prod_area_id
        self._prod_area = typing.cast(
            'ProductionArea',
            self.env.infstruct_mgr.lookup_subsystem_info(
                system_type=SimSystemTypes.PRODUCTION_AREA,
                lookup_val=self._prod_area_id,
            ),
        )

    def __repr__(self) -> str:
        return super().__repr__() + f' | ProductionAreaID: {self._prod_area_id}'

    def constant_sequence(
        self,
        order_time_source: Timedelta,
    ) -> Iterator[tuple[SystemID, SystemID, OrderTime]]:
        """Generates a constant sequence of job generation infos

        Parameters
        ----------
        stat_group_ids : Iterable[SystemID]
            target station group IDs
        order_time : Timedelta
            time for each order

        Yields
        -------
        tuple[SystemID, SystemID, Timedelta, Timedelta]
            prod area ID, target station group ID, processing time, setup time
        """
        # request StationGroupIDs by ProdAreaID in StationGroup database
        stat_group_db = self.env.infstruct_mgr.station_group_db
        filter_by_prod_area = stat_group_db.loc[
            stat_group_db['prod_area_id'] == self._prod_area_id, :
        ]
        stat_groups: list['StationGroup'] = filter_by_prod_area['station_group'].tolist()
        # stat_group_ids = filter_by_prod_area['station_group_id'].tolist()

        loggers.loads.debug('stat_groups: %s', stat_groups)

        # number of all processing stations in associated production area
        total_num_proc_stations: int = self._prod_area.num_assoc_proc_station

        loggers.loads.debug('total_num_proc_stations: %s', total_num_proc_stations)

        # order time equally distributed between all station within given ProductionArea
        # source distributes loads in round robin principle
        # order time for each station has to be the order time of the source
        # times the number of stations the source delivers to
        station_order_time = order_time_source * total_num_proc_stations

        loggers.loads.debug('station_order_time: %s', station_order_time)

        # generate endless sequence
        while True:
            # iterate over all StationGroups
            for stat_group in stat_groups:
                # generate job for each ProcessingStation in StationGroup
                for _ in range(stat_group.num_assoc_proc_station):
                    # generate random distribution of setup and processing time
                    setup_time_percentage = self._np_rnd_gen.uniform(low=0.1, high=0.8)
                    setup_time = setup_time_percentage * station_order_time
                    # round to next full minute
                    setup_time = self._dt_mgr.round_td_by_seconds(
                        td=setup_time, round_to_next_seconds=60
                    )
                    proc_time = station_order_time - setup_time
                    order_times = OrderTime(proc=proc_time, setup=setup_time)
                    # StationGroupID
                    stat_group_id = stat_group.system_id
                    yield (self._prod_area_id, stat_group_id, order_times)
