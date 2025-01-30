from __future__ import annotations

import enum
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NewType,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
)

from plotly.graph_objs._figure import Figure

if TYPE_CHECKING:
    from pyforcesim.constants import SimStatesCommon
    from pyforcesim.rl import agents
    from pyforcesim.simulation import environment as sim

T = TypeVar('T')


# ** logging
class LoggingLevels(enum.IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# ** common
PlotlyFigure: TypeAlias = Figure

FlattableObject: TypeAlias = (
    list['FlattableObject | Any']
    | tuple['FlattableObject | Any', ...]
    | set['FlattableObject | Any']
)

# ** simulation
SystemID = NewType('SystemID', int)
CustomID = NewType('CustomID', str)
LoadID = NewType('LoadID', int)
MonitorObjects: TypeAlias = (
    'sim.InfrastructureObject | sim.StorageLike | sim.Job | sim.Operation'
)
LoadObjects: TypeAlias = 'sim.Job | sim.Operation'
OrderPriority: TypeAlias = int
Infinite: TypeAlias = float
StateTimes: TypeAlias = dict[str, Timedelta]
SalabimTimeUnits: TypeAlias = Literal[
    'years',
    'weeks',
    'days',
    'hours',
    'minutes',
    'seconds',
    'milliseconds',
    'microseconds',
]
TimeTillDue: TypeAlias = Timedelta
DueDate: TypeAlias = Datetime


@dataclass(kw_only=True, slots=True, order=False)
class StatDistributionInfo:
    mean: float
    std: float


class DistributionParametersSet(TypedDict): ...


class DistUniformParameters(DistributionParametersSet):
    lower_bound: float
    upper_bound: float


class DistExpParameters(DistributionParametersSet):
    scale: float


@dataclass(match_args=False, eq=False, kw_only=True, slots=True)
class DistributionParameters:
    EXPONENTIAL: type[DistExpParameters] = DistExpParameters
    UNIFORM: type[DistUniformParameters] = DistUniformParameters


# class DistributionParameters(TypedDict): ...


# class DistUniformParameters(DistributionParameters):
#     lower_bound: float
#     upper_bound: float


# class DistExpParameters(DistributionParameters):
#     scale: float


class QueueLike(Protocol[T]):
    def env(self) -> sim.SimulationEnvironment: ...
    def custom_identifier(self) -> CustomID: ...
    def name(self) -> str: ...
    def pop(self, index: int | None = None) -> T: ...
    def append(self, item: T) -> None: ...
    def remove(self, item: Any) -> None: ...
    def as_list(self) -> list[T]: ...
    def __getitem__(self, index: int) -> T: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class OrderTimes:
    proc: Sequence[Timedelta]
    setup: Sequence[Timedelta]


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class OrderDates:
    starting_planned: Datetime | Sequence[Datetime | None] | None = field(default=None)
    ending_planned: Datetime | Sequence[Datetime | None] | None = field(default=None)


@dataclass(kw_only=True, slots=True)
class ExpectedArrivalTimes:
    ideal: float
    current: float


@dataclass(kw_only=True, slots=True)
class WIPInputTypes:
    factors: tuple[float, ...]
    times: tuple[Timedelta, ...]


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class JobGenerationInfo:
    custom_id: CustomID | None = field(default=None)
    execution_systems: Sequence[SystemID]
    station_groups: Sequence[SystemID]
    order_time: OrderTimes
    dates: OrderDates
    prio: OrderPriority | Sequence[OrderPriority | None] | None = field(default=None)
    current_state: SimStatesCommon


SourceSequence: TypeAlias = tuple[JobGenerationInfo, Timedelta]

# ** simulation environments
AgentType: TypeAlias = 'agents.AllocationAgent | agents.SequencingAgent'
EnvAgentConstructorReturn: TypeAlias = tuple[
    'sim.SimulationEnvironment',
    'agents.AllocationAgent | None',
    'agents.SequencingAgent | None',
]


class EnvBuilderFunc(Protocol):
    def __call__(
        self,
        sequencing: bool = ...,
        with_agent: bool = ...,
        validate: bool = ...,
        seed: int | None = ...,
        sim_dur_weeks: float = ...,
        num_station_groups: int = ...,
        num_machines: int = ...,
        variable_source_sequence: bool = ...,
        debug: bool = ...,
        seed_layout: int | None = ...,
        factor_WIP: float = ...,
        WIP_relative_target: Sequence[float] = ...,
        WIP_level_cycles: int = ...,
        WIP_relative_planned: float = ...,
        alpha: float = ...,
        buffer_size: int = ...,
    ) -> EnvAgentConstructorReturn: ...


class EnvBuilderAdditionalConfig(TypedDict):
    sim_dur_weeks: float
    factor_WIP: float | None
    WIP_relative_target: Sequence[float]
    WIP_level_cycles: int
    WIP_relative_planned: float
    alpha: float
    buffer_size: int


class BuilderFuncFamilies(enum.StrEnum):
    SINGLE_PRODUCTION_AREA = enum.auto()


@dataclass(slots=True, kw_only=True, eq=False)
class EnvGenerationInfo:
    num_station_groups: int
    num_machines: int
    variable_source_sequence: bool
    validate: bool


# ** agents
AgentTasks: TypeAlias = Literal['SEQ', 'ALLOC']


class AgentDecisionTypes(enum.StrEnum):
    ALLOC = enum.auto()
    SEQ = enum.auto()


# ** database
PandasDateColParseInfo: TypeAlias = dict[str, dict[str, bool]]
PandasDatetimeCols: TypeAlias = list[str]
PandasTimedeltaCols: TypeAlias = list[str]
# TODO check removal
DBColumnName: TypeAlias = str
DBColumnType: TypeAlias = str
DBColumnDeclaration: TypeAlias = dict[DBColumnName, DBColumnType]
SQLiteColumnDescription: TypeAlias = tuple[
    int,
    DBColumnName,
    DBColumnType,
    int,
    Any | None,
    int,
]


class ForeignKeyInfo(TypedDict):
    column: str
    ref_table: str
    ref_column: str


SysIDResource: TypeAlias = SystemID
LoadDistribution: TypeAlias = dict[SysIDResource, float]
