import enum
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
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
    LoadTimeRemainingPolicy,
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
from pyforcesim.types import (
    DistributionParameters,
    Infinite,
)
from pyforcesim.types import LoggingLevels as loglevel

# ** logging
LOG_FMT: Final[str] = ' %(asctime)s | pyfsim:%(module)s:%(levelname)s | %(message)s'
LOG_DATE_FMT: Final[str] = '%Y-%m-%d %H:%M:%S +0000'
LOGGING_ENABLED: Final[bool] = True
LOGGING_TO_FILE: Final[bool] = False
LOGGING_FILE_SIZE: Final[int] = 10485760  # in bytes
LOGGING_LEVEL_STD_OUT: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_FILE: Final[loglevel] = loglevel.DEBUG
LOGGING_LEVEL_BASE: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_ENV: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_GYM_ENV: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_ENV_BUILDER: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_DISPATCHER: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_INFSTRCT: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SOURCES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SINKS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_PRODSTATIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_JOBS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_OPERATIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_BUFFERS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_QUEUES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_LOADS: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_MONITORS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_AGENTS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_CONDITIONS: Final[loglevel] = loglevel.DEBUG
LOGGING_LEVEL_POLICIES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_DB: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_DIST: Final[loglevel] = loglevel.DEBUG


# ** common
# infinity
INF: Final[Infinite] = float('inf')
DEFAULT_SEED: Final[int] = 42
EPSILON: Final[float] = 1e-8
ROUNDING_PRECISION: Final[int] = 6


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
SLACK_INIT_AS_UPPER_BOUND: Final[bool] = True
# value to use as slack if initial value is not set as upper bound
SLACK_OVERWRITE_UPPER_BOUND: Final[Timedelta] = Timedelta(hours=1)
# only takes effect if initial slack used as upper bound
SLACK_USE_THRESHOLD_UPPER: Final[bool] = True
SLACK_THRESHOLD_UPPER: Final[Timedelta] = Timedelta(hours=2)
SLACK_THRESHOLD_LOWER: Final[Timedelta] = Timedelta()


# ** database
DB_HANDLE: Final[str] = 'sqlite:///:memory:'
DB_ECHO: Final[bool] = False
DB_ROOT: Final[str] = 'databases'


# ** simulation
# indicator how much workload can be processed per day
# since a day has 24 hours each infrastructure object can process
# 24 hours of workload per day at the maximum
MAX_PROCESSING_CAPACITY: Final[Timedelta] = Timedelta(hours=24)
MAX_LOGICAL_QUEUE_SIZE: Final[int] = 60
SEQUENCING_WAITING_TIME: Final[Timedelta] = Timedelta(minutes=15)
SOURCE_GENERATION_WAITING_TIME: Final[Timedelta] = Timedelta(minutes=5)


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
PROCESSING_PROPERTIES: Final[frozenset[SimStatesCommon]] = frozenset(
    [
        SimStatesCommon.PROCESSING,
        SimStatesCommon.SETUP,
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
    LOGICAL_QUEUE = enum.auto()


class JobGeneration(enum.StrEnum):
    RANDOM = enum.auto()
    SEQUENTIAL = enum.auto()


# ** distribution types
class StatisticalDistributionsSupported(enum.StrEnum):
    EXPONENTIAL = enum.auto()
    UNIFORM = enum.auto()


DISTRIBUTION_PARAMETERS: Final[DistributionParameters] = DistributionParameters()


# ** policies
POLICIES: Final[dict[str, type[Policy]]] = {
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
    'LOAD_TIME_REMAINING': LoadTimeRemainingPolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
}

POLICIES_SEQ: Final[dict[str, type[GeneralPolicy | SequencingPolicy]]] = {
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

POLICIES_ALLOC: Final[dict[str, type[GeneralPolicy | AllocationPolicy]]] = {
    'AGENT': AgentPolicy,
    'LOAD_TIME': LoadTimePolicy,
    'LOAD_TIME_REMAINING': LoadTimeRemainingPolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
    'RANDOM': RandomPolicy,
}
