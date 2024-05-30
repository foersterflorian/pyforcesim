from typing import Final

from pyforcesim.simulation.policies import (
    AgentPolicy,
    FIFOPolicy,
    LIFOPolicy,
    LoadJobsPolicy,
    LoadTimePolicy,
    LPTPolicy,
    LSTPolicy,
    PriorityPolicy,
    RandomPolicy,
    SPTPolicy,
    SSTPolicy,
    UtilisationPolicy,
)
from pyforcesim.types import Infinite

# infinity
INF: Final[Infinite] = float('inf')
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
POLICIES: Final[dict[str, type]] = {
    'AGENT': AgentPolicy,
    'FIFO': FIFOPolicy,
    'LIFO': LIFOPolicy,
    'SPT': SPTPolicy,
    'LPT': LPTPolicy,
    'SST': SSTPolicy,
    'LST': LSTPolicy,
    'PRIORITY': PriorityPolicy,
    'RANDOM': RandomPolicy,
    'LOAD_TIME': LoadTimePolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
}
POLICIES_SEQ: Final[dict[str, type]] = {
    'AGENT': AgentPolicy,
    'FIFO': FIFOPolicy,
    'LIFO': LIFOPolicy,
    'SPT': SPTPolicy,
    'LPT': LPTPolicy,
    'SST': SSTPolicy,
    'LST': LSTPolicy,
    'PRIORITY': PriorityPolicy,
    'RANDOM': RandomPolicy,
}
POLICIES_ALLOC: Final[dict[str, type]] = {
    'AGENT': AgentPolicy,
    'LOAD_TIME': LoadTimePolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
    'RANDOM': RandomPolicy,
}
