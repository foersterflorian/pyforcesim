from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from operator import attrgetter
from random import Random
from typing import TYPE_CHECKING, Never, TypeVar, cast
from typing_extensions import override

from pyforcesim.types import SystemID

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        Job,
        Operation,
        ProcessingStation,
        StationGroup,
    )

T = TypeVar('T')


class Policy(ABC):
    def __str__(self) -> str:
        return f'Policy({self.name})'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(
        self,
        items: Sequence[T],
    ) -> T: ...


class GeneralPolicy(Policy):
    pass


class AllocationPolicy(Policy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.last_by_stat_group: dict[SystemID, SystemID] = {}

    @staticmethod
    def get_station_group(
        proc: ProcessingStation,
    ) -> StationGroup:
        return cast('StationGroup', proc.supersystems_as_tuple()[0])

    def load_balancing(
        self,
        station_group: StationGroup,
        min_val_procs: Sequence[ProcessingStation],
    ) -> ProcessingStation:
        """re-assigns the processing station if it was already chosen before
        Background: Processing stations which have the same value for a given
        property are chosen based on their index in the given sequence. This can lead
        to a situation where the same processing station is chosen multiple times in a
        row. This methods prevents this by re-assigning the processing station to the next
        one with the same value for the given property.

        Parameters
        ----------
        station_group : StationGroup
            station group of the processing stations
        min_val_procs : Sequence[ProcessingStation]
            sequence of processing stations with the same value for the given property

        Returns
        -------
        ProcessingStation
            processing station which should be chosen
        """
        if len(min_val_procs) == 1:
            proc_min = min_val_procs[0]
            self.last_by_stat_group[station_group.system_id] = proc_min.system_id
        else:
            for proc in min_val_procs:
                if (
                    station_group.system_id not in self.last_by_stat_group
                    or self.last_by_stat_group[station_group.system_id] != proc.system_id
                ):
                    self.last_by_stat_group[station_group.system_id] = proc.system_id
                    proc_min = proc
                    break
                elif self.last_by_stat_group[station_group.system_id] == proc.system_id:
                    continue

        return proc_min


class SequencingPolicy(Policy):
    pass


# ** General


class RandomPolicy(GeneralPolicy):
    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._rng = Random(seed)

    @property
    def rng(self) -> Random:
        return self._rng

    @override
    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return self.rng.choice(items)


class FIFOPolicy(GeneralPolicy):
    @override
    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return items[0]


class LIFOPolicy(GeneralPolicy):
    @override
    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return items[-1]


class AgentPolicy(GeneralPolicy):
    @override
    def apply(
        self,
        _: Sequence[T],
    ) -> Never:
        raise NotImplementedError('AgentPolicy implemented in different way.')


# ** Sequencing


class SPTPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_proc_time'))


class LPTPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_proc_time'))


class SSTPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_setup_time'))


class LSTPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_setup_time'))


class PriorityPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('prio'))


# !! currently only op-wise
class EDDPolicy(SequencingPolicy):
    @override
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        ops = tuple((job.current_op for job in items))
        assert all((op is not None for op in ops)), 'at least one current operation is None'
        ops = cast(tuple['Operation', ...], ops)
        target_op = min(ops, key=attrgetter('time_planned_ending'))

        return target_op.job


# ** Allocation


class UtilisationPolicy(AllocationPolicy):
    @override
    def apply(
        self,
        items: Iterable[ProcessingStation],
    ) -> ProcessingStation:
        proc_min = min(items, key=attrgetter('stat_monitor.utilisation'))
        station_group = self.get_station_group(proc_min)
        min_val_procs = [
            proc
            for proc in items
            if (proc.stat_monitor.utilisation == proc_min.stat_monitor.utilisation)
        ]
        proc_min = self.load_balancing(station_group, min_val_procs)

        return proc_min


class LoadTimePolicy(AllocationPolicy):
    @override
    def apply(
        self,
        items: Iterable[ProcessingStation],
    ) -> ProcessingStation:
        proc_min = min(items, key=attrgetter('stat_monitor.WIP_load_time'))
        station_group = self.get_station_group(proc_min)
        min_val_procs = [
            proc
            for proc in items
            if (proc.stat_monitor.WIP_load_time == proc_min.stat_monitor.WIP_load_time)
        ]
        proc_min = self.load_balancing(station_group, min_val_procs)

        return proc_min


class LoadTimeRemainingPolicy(AllocationPolicy):
    @override
    def apply(
        self,
        items: Iterable[ProcessingStation],
    ) -> ProcessingStation:
        proc_min = min(items, key=attrgetter('stat_monitor.WIP_load_time_remaining'))
        station_group = self.get_station_group(proc_min)
        min_val_procs = [
            proc
            for proc in items
            if (
                proc.stat_monitor.WIP_load_time_remaining
                == proc_min.stat_monitor.WIP_load_time_remaining
            )
        ]
        proc_min = self.load_balancing(station_group, min_val_procs)

        return proc_min


class LoadJobsPolicy(AllocationPolicy):
    @override
    def apply(
        self,
        items: Iterable[ProcessingStation],
    ) -> ProcessingStation:
        proc_min = min(items, key=attrgetter('stat_monitor.WIP_load_num_jobs'))
        station_group = self.get_station_group(proc_min)
        min_val_procs = [
            proc
            for proc in items
            if (
                proc.stat_monitor.WIP_load_num_jobs == proc_min.stat_monitor.WIP_load_num_jobs
            )
        ]
        proc_min = self.load_balancing(station_group, min_val_procs)

        return proc_min


class RoundRobinPolicy(AllocationPolicy):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.last_by_stat_group: dict[SystemID, ProcessingStation] = {}

    @override
    def apply(
        self,
        items: Sequence[ProcessingStation],
    ) -> Never:
        raise NotImplementedError('RoundRobinPolicy not implemented yet.')
