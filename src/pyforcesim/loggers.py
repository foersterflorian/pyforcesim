"""provides logger objects for the pyforcesim package"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path

from pyforcesim import common
from pyforcesim.constants import (
    LOG_DATE_FMT,
    LOG_FMT,
    LOGGING_ENABLED,
    LOGGING_FILE_SIZE,
    LOGGING_LEVEL_AGENTS,
    LOGGING_LEVEL_BASE,
    LOGGING_LEVEL_BUFFERS,
    LOGGING_LEVEL_CONDITIONS,
    LOGGING_LEVEL_DB,
    LOGGING_LEVEL_DISPATCHER,
    LOGGING_LEVEL_ENV,
    LOGGING_LEVEL_ENV_BUILDER,
    LOGGING_LEVEL_FILE,
    LOGGING_LEVEL_GYM_ENV,
    LOGGING_LEVEL_INFSTRCT,
    LOGGING_LEVEL_JOBS,
    LOGGING_LEVEL_LOADS,
    LOGGING_LEVEL_MONITORS,
    LOGGING_LEVEL_OPERATIONS,
    LOGGING_LEVEL_POLICIES,
    LOGGING_LEVEL_PRODSTATIONS,
    LOGGING_LEVEL_QUEUES,
    LOGGING_LEVEL_SINKS,
    LOGGING_LEVEL_SOURCES,
    LOGGING_LEVEL_STD_OUT,
    LOGGING_TO_FILE,
)

base = logging.getLogger('pyforcesim')
base.setLevel(LOGGING_LEVEL_BASE)
# formatters
formatter = logging.Formatter(LOG_FMT, LOG_DATE_FMT)
formatter.converter = time.gmtime
# handlers STDERR
handler_null = logging.NullHandler()
base.addHandler(handler_null)
handler_stdout = logging.StreamHandler(stream=sys.stdout)  # IPython compatibility
handler_stdout.setLevel(LOGGING_LEVEL_STD_OUT)
handler_stdout.setFormatter(formatter)

if LOGGING_ENABLED:
    base.removeHandler(handler_null)
    base.addHandler(handler_stdout)

if LOGGING_ENABLED and LOGGING_TO_FILE:
    timestamp = common.get_timestamp(with_time=False)
    logging_pth = Path.cwd() / f'logs_{timestamp}.txt'
    handler_file = logging.handlers.RotatingFileHandler(
        logging_pth,
        maxBytes=LOGGING_FILE_SIZE,
        backupCount=2,
    )
    handler_file.setLevel(LOGGING_LEVEL_FILE)
    handler_file.setFormatter(formatter)
    base.addHandler(handler_file)


pyf_env = logging.getLogger('pyforcesim.sim_env')
pyf_env.setLevel(LOGGING_LEVEL_ENV)
dispatcher = logging.getLogger('pyforcesim.dispatcher')
dispatcher.setLevel(LOGGING_LEVEL_DISPATCHER)
infstrct = logging.getLogger('pyforcesim.infstrct')
infstrct.setLevel(LOGGING_LEVEL_INFSTRCT)
sources = logging.getLogger('pyforcesim.sources')
sources.setLevel(LOGGING_LEVEL_SOURCES)
sinks = logging.getLogger('pyforcesim.sinks')
sinks.setLevel(LOGGING_LEVEL_SINKS)
prod_stations = logging.getLogger('pyforcesim.prodStations')
prod_stations.setLevel(LOGGING_LEVEL_PRODSTATIONS)
buffers = logging.getLogger('pyforcesim.buffers')
buffers.setLevel(LOGGING_LEVEL_BUFFERS)
queues = logging.getLogger('pyforcesim.queues')
queues.setLevel(LOGGING_LEVEL_QUEUES)
loads = logging.getLogger('pyforcesim.loads')
loads.setLevel(LOGGING_LEVEL_LOADS)
monitors = logging.getLogger('pyforcesim.monitors')
monitors.setLevel(LOGGING_LEVEL_MONITORS)
agents = logging.getLogger('pyforcesim.agents')
agents.setLevel(LOGGING_LEVEL_AGENTS)
conditions = logging.getLogger('pyforcesim.conditions')
conditions.setLevel(LOGGING_LEVEL_CONDITIONS)
policies = logging.getLogger('pyforcesim.policies')
policies.setLevel(LOGGING_LEVEL_POLICIES)
databases = logging.getLogger('pyforcesim.databases')
databases.setLevel(LOGGING_LEVEL_DB)

jobs = logging.getLogger('pyforcesim.jobs')
jobs.setLevel(LOGGING_LEVEL_JOBS)
operations = logging.getLogger('pyforcesim.operations')
operations.setLevel(LOGGING_LEVEL_OPERATIONS)

gym_env = logging.getLogger('pyforcesime.gym_env')
gym_env.setLevel(LOGGING_LEVEL_GYM_ENV)
env_builder = logging.getLogger('pyforcesime.env_builder')
env_builder.setLevel(LOGGING_LEVEL_ENV_BUILDER)
