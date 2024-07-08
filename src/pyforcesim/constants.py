import datetime
import enum
from datetime import datetime as Datetime
from datetime import timezone as Timezone
from typing import Final
from zoneinfo import ZoneInfo

from pyforcesim.simulation.policies import (
    AgentPolicy,
    AllocationPolicy,
    FIFOPolicy,
    GeneralPolicy,
    LIFOPolicy,
    LoadJobsPolicy,
    LoadTimePolicy,
    LPTPolicy,
    LSTPolicy,
    Policy,
    PriorityPolicy,
    RandomPolicy,
    SequencingPolicy,
    SPTPolicy,
    SSTPolicy,
    UtilisationPolicy,
)
from pyforcesim.types import Infinite
from pyforcesim.types import LoggingLevels as loglevel

# ** logging
LOGGING_TO_FILE: Final[bool] = False
LOGGING_LEVEL_BASE: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_ENV: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_GYM_ENV: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_DISPATCHER: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_INFSTRCT: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SOURCES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SINKS: Final[loglevel] = loglevel.ERROR
LOGGING_LEVEL_PRODSTATIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_JOBS: Final[loglevel] = loglevel.ERROR
LOGGING_LEVEL_OPERATIONS: Final[loglevel] = loglevel.ERROR
LOGGING_LEVEL_BUFFERS: Final[loglevel] = loglevel.ERROR
LOGGING_LEVEL_LOADS: Final[loglevel] = loglevel.ERROR
LOGGING_LEVEL_MONITORS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_AGENTS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_CONDITIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_DB: Final[loglevel] = loglevel.ERROR


# ** common
# infinity
INF: Final[Infinite] = float('inf')


# ** dates and times
class TimeUnitsDatetime(enum.StrEnum):
    YEAR = enum.auto()
    MONTH = enum.auto()
    DAY = enum.auto()
    HOUR = enum.auto()
    MINUTE = enum.auto()
    SECOND = enum.auto()
    MICROSECOND = enum.auto()


class TimeUnitsTimedelta(enum.StrEnum):
    WEEKS = enum.auto()
    DAYS = enum.auto()
    HOURS = enum.auto()
    MINUTES = enum.auto()
    SECONDS = enum.auto()
    MILLISECONDS = enum.auto()
    MICROSECONDS = enum.auto()


TIMEZONE_CEST: Final[ZoneInfo] = ZoneInfo('Europe/Berlin')
TIMEZONE_UTC: Final[Timezone] = Timezone.utc
DEFAULT_DATETIME: Final[Datetime] = Datetime(1970, 1, 1, tzinfo=TIMEZONE_UTC)


# ** database
DB_HANDLE: Final[str] = 'sqlite:///:memory:'
DB_ECHO: Final[bool] = False
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
DB_INJECTION_PATTERN: Final[str] = (
    r'(^[_0-9]+)|[^\w ]|(true|false|select|where|drop|delete|create)'
)


# ** simulation
class SimResourceTypes(enum.StrEnum):
    MACHINE = enum.auto()
    STORAGE = enum.auto()
    BUFFER = enum.auto()
    SOURCE = enum.auto()
    SINK = enum.auto()
    ASSEMBLY = enum.auto()
    PROCESSING_STATION = enum.auto()


class SimStatesCommon(enum.StrEnum):
    INIT = enum.auto()
    FINISH = enum.auto()
    TEMP = enum.auto()
    IDLE = enum.auto()
    PROCESSING = enum.auto()
    SETUP = enum.auto()
    PAUSED = enum.auto()
    BLOCKED = enum.auto()
    FAILED = enum.auto()


class SimStatesAvailability(enum.StrEnum):
    IDLE = enum.auto()


class SimStatesStorage(enum.StrEnum):
    INIT = enum.auto()
    FINISH = enum.auto()
    TEMP = enum.auto()
    FAILED = enum.auto()
    PAUSED = enum.auto()
    FULL = enum.auto()
    EMPTY = enum.auto()
    INTERMEDIATE = enum.auto()


UTIL_PROPERTIES: Final[frozenset[SimStatesCommon]] = frozenset(
    [
        SimStatesCommon.PROCESSING,
        SimStatesCommon.SETUP,
        SimStatesCommon.PAUSED,
    ]
)
HELPER_STATES: Final[frozenset[SimStatesCommon]] = frozenset(
    [
        SimStatesCommon.INIT,
        SimStatesCommon.FINISH,
        SimStatesCommon.TEMP,
    ]
)


class SimSystemTypes(enum.StrEnum):
    PRODUCTION_AREA = enum.auto()
    STATION_GROUP = enum.auto()
    RESOURCE = enum.auto()


class JobGeneration(enum.StrEnum):
    RANDOM = enum.auto()
    SEQUENTIAL = enum.auto()


# ** policies
POLICIES: Final[dict[str, Policy]] = {
    'AGENT': AgentPolicy(),
    'FIFO': FIFOPolicy(),
    'LIFO': LIFOPolicy(),
    'SPT': SPTPolicy(),
    'LPT': LPTPolicy(),
    'SST': SSTPolicy(),
    'LST': LSTPolicy(),
    'PRIORITY': PriorityPolicy(),
    'RANDOM': RandomPolicy(),
    'LOAD_TIME': LoadTimePolicy(),
    'LOAD_JOBS': LoadJobsPolicy(),
    'UTILISATION': UtilisationPolicy(),
}

POLICIES_SEQ: Final[dict[str, GeneralPolicy | SequencingPolicy]] = {
    'AGENT': AgentPolicy(),
    'FIFO': FIFOPolicy(),
    'LIFO': LIFOPolicy(),
    'SPT': SPTPolicy(),
    'LPT': LPTPolicy(),
    'SST': SSTPolicy(),
    'LST': LSTPolicy(),
    'PRIORITY': PriorityPolicy(),
    'RANDOM': RandomPolicy(),
}

POLICIES_ALLOC: Final[dict[str, GeneralPolicy | AllocationPolicy]] = {
    'AGENT': AgentPolicy(),
    'LOAD_TIME': LoadTimePolicy(),
    'LOAD_JOBS': LoadJobsPolicy(),
    'UTILISATION': UtilisationPolicy(),
    'RANDOM': RandomPolicy(),
}
