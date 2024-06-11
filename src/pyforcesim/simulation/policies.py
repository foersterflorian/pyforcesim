from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from operator import attrgetter
from random import Random
from typing import TYPE_CHECKING, Never, TypeVar, cast

from pyforcesim.types import SystemID

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import (
        Job,
        ProcessingStation,
        StationGroup,
    )

T = TypeVar('T')


class Policy(ABC):
    def __str__(self) -> str:
        return f'Policy({self.__class__.__name__})'

    def __repr__(self) -> str:
        return self.__str__()

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

    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return self.rng.choice(items)


class FIFOPolicy(GeneralPolicy):
    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return items[0]


class LIFOPolicy(GeneralPolicy):
    def apply(
        self,
        items: Sequence[T],
    ) -> T:
        return items[-1]


class AgentPolicy(GeneralPolicy):
    def apply(
        self,
        _: Sequence[T],
    ) -> Never:
        raise NotImplementedError('AgentPolicy implemented in different way.')


# ** Sequencing


class SPTPolicy(SequencingPolicy):
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_proc_time'))


class LPTPolicy(SequencingPolicy):
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_proc_time'))


class SSTPolicy(SequencingPolicy):
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_setup_time'))


class LSTPolicy(SequencingPolicy):
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_setup_time'))


class PriorityPolicy(SequencingPolicy):
    def apply(
        self,
        items: Sequence[Job],
    ) -> Job:
        return max(items, key=attrgetter('prio'))


# ** Allocation


class UtilisationPolicy(AllocationPolicy):
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


class LoadJobsPolicy(AllocationPolicy):
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

    def apply(
        self,
        items: Sequence[ProcessingStation],
    ) -> Never:
        raise NotImplementedError('RoundRobinPolicy not implemented yet.')
