from typing import Any, Literal, NewType, TypeAlias, TypedDict

from plotly.graph_objs._figure import Figure

# ** common
LoggingLevels: TypeAlias = Literal[
    'DEBUG',
    'INFO',
    'WARNING',
    'ERROR',
    'CRITICAL',
]
PlotlyFigure: TypeAlias = Figure

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
