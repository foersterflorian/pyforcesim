from typing import Any, Final, Literal, NewType, TypeAlias, TypedDict

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

# infinity
INF: Final[Infinite] = float('inf')

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


DB_ROOT: Final[str] = 'databases'
DB_DATA_TYPES: Final[set[str]] = {
    'INTEGER',
    'REAL',
    'BOOLEAN',
    'TEXT',
    'BLOB',
    'DATE',
    'DATETIME',
    'TIMEDELTA',
}
# SQLite supports more column constraints than these,
# but only these are currently supported by the database module
DB_SUPPORTED_COL_CONSTRAINTS: Final[frozenset[str]] = frozenset(
    [
        'NOT NULL',
        'UNIQUE',
        'PRIMARY KEY',
    ]
)
