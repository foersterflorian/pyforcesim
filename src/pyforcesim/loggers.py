"""provides logger objects for the pyforcesim package"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Final

from pyforcesim.constants import (
    LOGGING_LEVEL_AGENTS,
    LOGGING_LEVEL_BASE,
    LOGGING_LEVEL_BUFFERS,
    LOGGING_LEVEL_CONDITIONS,
    LOGGING_LEVEL_DB,
    LOGGING_LEVEL_DISPATCHER,
    LOGGING_LEVEL_ENV,
    LOGGING_LEVEL_ENV_BUILDER,
    LOGGING_LEVEL_GYM_ENV,
    LOGGING_LEVEL_INFSTRCT,
    LOGGING_LEVEL_JOBS,
    LOGGING_LEVEL_LOADS,
    LOGGING_LEVEL_MONITORS,
    LOGGING_LEVEL_OPERATIONS,
    LOGGING_LEVEL_POLICIES,
    LOGGING_LEVEL_PRODSTATIONS,
    LOGGING_LEVEL_SINKS,
    LOGGING_LEVEL_SOURCES,
    LOGGING_TO_FILE,
)

# IPython compatibility with stdout
logging.Formatter.converter = time.gmtime
LOG_FMT: Final[str] = ' %(asctime)s | pyfsim:%(module)s:%(levelname)s | %(message)s'
LOG_DATE_FMT: Final[str] = '%Y-%m-%d %H:%M:%S +0000'
if LOGGING_TO_FILE:
    logging_pth = Path.cwd() / 'logs.txt'
    if logging_pth.exists():
        os.remove(logging_pth)
    logging.basicConfig(
        filename=logging_pth,
        filemode='a',
        format=LOG_FMT,
        datefmt=LOG_DATE_FMT,
    )
else:
    logging.basicConfig(
        stream=sys.stdout,
        format=LOG_FMT,
        datefmt=LOG_DATE_FMT,
    )

base = logging.getLogger('pyforcesim.base')
base.setLevel(LOGGING_LEVEL_BASE)
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
