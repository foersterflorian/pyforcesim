import enum
from typing import Any, Literal, NewType, TypeAlias, TypedDict

from plotly.graph_objs._figure import Figure


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
Infinite: TypeAlias = float


# ** agents
AgentTasks: TypeAlias = Literal['SEQ', 'ALLOC']


# ** database
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
