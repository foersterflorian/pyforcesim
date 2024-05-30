from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from operator import attrgetter
from random import Random
from typing import TYPE_CHECKING, Never, TypeVar

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import Job, ProcessingStation

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
    pass


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
        items: list[T],
    ) -> T:
        return items[0]


class LIFOPolicy(GeneralPolicy):
    def apply(
        self,
        items: list[T],
    ) -> T:
        return items[-1]


class AgentPolicy(GeneralPolicy):
    def apply(self) -> Never:
        raise NotImplementedError('AgentPolicy implemented in different way.')


# ** Sequencing


class SPTPolicy(SequencingPolicy):
    def apply(
        self,
        items: list[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_proc_time'))


class LPTPolicy(SequencingPolicy):
    def apply(
        self,
        items: list[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_proc_time'))


class SSTPolicy(SequencingPolicy):
    def apply(
        self,
        items: list[Job],
    ) -> Job:
        return min(items, key=attrgetter('current_setup_time'))


class LSTPolicy(SequencingPolicy):
    def apply(
        self,
        items: list[Job],
    ) -> Job:
        return max(items, key=attrgetter('current_setup_time'))


class PriorityPolicy(SequencingPolicy):
    def apply(
        self,
        items: list[Job],
    ) -> Job:
        return max(items, key=attrgetter('prio'))


# ** Allocation


class UtilisationPolicy(AllocationPolicy):
    def apply(
        self,
        items: Sequence[ProcessingStation],
    ) -> ProcessingStation:
        return min(items, key=attrgetter('stat_monitor.WIP_load_time'))


class LoadTimePolicy(AllocationPolicy):
    def apply(
        self,
        items: Sequence[ProcessingStation],
    ) -> ProcessingStation:
        return min(items, key=attrgetter('stat_monitor.WIP_load_time'))


class LoadJobsPolicy(AllocationPolicy):
    def apply(
        self,
        items: Sequence[ProcessingStation],
    ) -> ProcessingStation:
        return min(items, key=attrgetter('stat_monitor.WIP_load_num_jobs'))
