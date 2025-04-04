import enum
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from datetime import timezone as Timezone
from typing import Final
from zoneinfo import ZoneInfo

from pyforcesim.config import CFG
from pyforcesim.simulation.policies import (
    AgentPolicy,
    AllocationPolicy,
    EDDPolicy,
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
LOGGING_ENABLED: Final[bool] = CFG.lib.logging.enabled
LOGGING_TO_FILE: Final[bool] = CFG.lib.logging.file
LOGGING_FILE_SIZE: Final[int] = 10485760  # in bytes
LOGGING_LEVEL_STD_OUT: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_FILE: Final[loglevel] = loglevel.DEBUG
LOGGING_LEVEL_BASE: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_ENV: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_GYM_ENV: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_ENV_BUILDER: Final[loglevel] = loglevel.INFO
LOGGING_LEVEL_DISPATCHER: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_INFSTRCT: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SOURCES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_SINKS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_PRODSTATIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_JOBS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_OPERATIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_BUFFERS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_QUEUES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_LOADS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_MONITORS: Final[loglevel] = loglevel.DEBUG
LOGGING_LEVEL_AGENTS: Final[loglevel] = loglevel.DEBUG
LOGGING_LEVEL_CONDITIONS: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_POLICIES: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_DB: Final[loglevel] = loglevel.WARNING
LOGGING_LEVEL_DIST: Final[loglevel] = loglevel.WARNING

# ** config
# ** GymEnv
CFG_SIM_DUR_WEEKS: Final[int] = CFG.lib.gym_env.sim_dur_weeks
CFG_BUFFER_SIZE: Final[int] = CFG.lib.gym_env.buffer_size
CFG_JOB_POOL_SIZE_MIN: Final[int] = CFG.lib.gym_env.job_pool_size_min
CFG_JOB_POOL_SIZE_MAX: Final[int] = CFG.lib.gym_env.job_pool_size_max
CFG_DISPATCHER_SEQ_RULE: Final[str] = CFG.lib.gym_env.dispatcher_seq_rule
CFG_DISPATCHER_ALLOC_RULE: Final[str] = CFG.lib.gym_env.dispatcher_alloc_rule
# ** GymEnv: WIP
CFG_FACTOR_WIP: Final[float | None] = CFG.lib.gym_env.WIP.factor_WIP
CFG_USE_WIP_TARGETS: Final[bool] = CFG.lib.gym_env.WIP.use_WIP_targets
CFG_WIP_RELATIVE_TARGETS: Final[tuple[float, ...]] = CFG.lib.gym_env.WIP.WIP_relative_targets
CFG_WIP_LEVEL_CYCLES: Final[int] = CFG.lib.gym_env.WIP.WIP_level_cycles
CFG_WIP_RELATIVE_PLANNED: Final[float] = CFG.lib.gym_env.WIP.WIP_relative_planned
CFG_ALPHA: Final[float] = CFG.lib.gym_env.WIP.alpha
# ** GymEnv: WIP targets
CFG_WIP_TARGET_MIN: Final[float] = CFG.lib.gym_env.WIP_targets.min
CFG_WIP_TARGET_MAX: Final[float] = CFG.lib.gym_env.WIP_targets.max
CFG_WIP_TARGET_NUM_LEVELS: Final[int] = CFG.lib.gym_env.WIP_targets.number_WIP_levels


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
SLACK_INIT_AS_UPPER_BOUND: Final[bool] = CFG.lib.sim.slack.init_as_upper_bound
# vvv - only takes effect if initial slack used as upper bound - vvv
SLACK_USE_THRESHOLD_UPPER: Final[bool] = CFG.lib.sim.slack.use_threshold_upper
SLACK_THRESHOLD_UPPER: Final[Timedelta] = Timedelta(hours=CFG.lib.sim.slack.threshold_upper)
# vvv - slack: default values and ranges - vvv
SLACK_DEFAULT_LOWER_BOUND: Final[Timedelta] = Timedelta(
    hours=CFG.lib.sim.slack.default_lower_bound
)
SLACK_MIN_RANGE: Final[Timedelta] = Timedelta(hours=CFG.lib.sim.slack.min_range)
SLACK_MAX_RANGE: Final[Timedelta] = Timedelta(hours=CFG.lib.sim.slack.max_range)
# vvv - value to use as slack if initial value is not set as upper bound - vvv
SLACK_OVERWRITE_UPPER_BOUND: Final[Timedelta] = Timedelta(
    hours=CFG.lib.sim.slack.overwrite_upper_bound
)
# slack adaption
SLACK_ADAPTION: Final[bool] = CFG.lib.sim.slack.adaption
# !! vvv - currently without effect - vvv
SLACK_ADAPTION_MIN_UPPER_BOUND: Final[Timedelta] = Timedelta(
    hours=CFG.lib.sim.slack.adaption_min_upper_bound
)
SLACK_ADAPTION_MIN_LOWER_BOUND: Final[Timedelta] = Timedelta(
    hours=CFG.lib.sim.slack.adaption_min_lower_bound
)


# ** database
DB_HANDLE: Final[str] = 'sqlite:///:memory:'
DB_ECHO: Final[bool] = False
DB_ROOT: Final[str] = 'databases'


# ** simulation
# WIP setter interval: how many full cycles (each level used once) should occur
WIP_LEVELS_FULL_CYCLES: Final[int] = 7
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
POLICIES_TYPE: Final[dict[str, type[Policy]]] = {
    'AGENT': AgentPolicy,
    'FIFO': FIFOPolicy,
    'LIFO': LIFOPolicy,
    'SPT': SPTPolicy,
    'LPT': LPTPolicy,
    'SST': SSTPolicy,
    'LST': LSTPolicy,
    'EDD': EDDPolicy,
    'PRIORITY': PriorityPolicy,
    'RANDOM': RandomPolicy,
    'LOAD_TIME': LoadTimePolicy,
    'LOAD_TIME_REMAINING': LoadTimeRemainingPolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
}

POLICIES_SEQ_TYPE: Final[dict[str, type[GeneralPolicy | SequencingPolicy]]] = {
    'AGENT': AgentPolicy,
    'FIFO': FIFOPolicy,
    'LIFO': LIFOPolicy,
    'SPT': SPTPolicy,
    'LPT': LPTPolicy,
    'SST': SSTPolicy,
    'LST': LSTPolicy,
    'EDD': EDDPolicy,
    'PRIORITY': PriorityPolicy,
    'RANDOM': RandomPolicy,
}

POLICIES_ALLOC_TYPE: Final[dict[str, type[GeneralPolicy | AllocationPolicy]]] = {
    'AGENT': AgentPolicy,
    'LOAD_TIME': LoadTimePolicy,
    'LOAD_TIME_REMAINING': LoadTimeRemainingPolicy,
    'LOAD_JOBS': LoadJobsPolicy,
    'UTILISATION': UtilisationPolicy,
    'RANDOM': RandomPolicy,
}

POLICIES: Final[frozenset[str]] = frozenset(POLICIES_TYPE.keys())
POLICIES_SEQ: Final[frozenset[str]] = frozenset(POLICIES_SEQ_TYPE.keys())
POLICIES_ALLOC: Final[frozenset[str]] = frozenset(POLICIES_ALLOC_TYPE.keys())
