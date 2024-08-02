from __future__ import annotations

import enum
from collections.abc import Sequence
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
)

from plotly.graph_objs._figure import Figure

if TYPE_CHECKING:
    from pyforcesim.constants import SimStatesCommon
    from pyforcesim.rl import agents
    from pyforcesim.simulation import environment as sim


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


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class OrderTimes:
    proc: Sequence[Timedelta]
    setup: Sequence[Timedelta]


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class OrderDates:
    starting_planned: Datetime | Sequence[Datetime | None] | None = field(default=None)
    ending_planned: Datetime | Sequence[Datetime | None] | None = field(default=None)


@dataclass(kw_only=True, slots=True, eq=False, match_args=False)
class JobGenerationInfo:
    custom_id: CustomID | None = field(default=None)
    execution_systems: Sequence[SystemID]
    station_groups: Sequence[SystemID | None] | None = field(default=None)
    order_time: OrderTimes
    dates: OrderDates
    prio: OrderPriority | Sequence[OrderPriority | None] | None = field(default=None)
    current_state: SimStatesCommon


# ** simulation environments
class EnvBuilderFunc(Protocol):
    def __call__(
        self,
        with_agent: bool = ...,
        seed: int | None = ...,
    ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]: ...


# ** agents
AgentTasks: TypeAlias = Literal['SEQ', 'ALLOC']


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
