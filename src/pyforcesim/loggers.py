"""provides logger objects for the pyforcesim package"""

import logging
import sys
from typing import Final

from pyforcesim.types import LoggingLevels

# IPython compatibility
logging.basicConfig(stream=sys.stdout)

LOGGING_LEVEL_BASE: Final[LoggingLevels] = 'DEBUG'
LOGGING_LEVEL_ENV: Final[LoggingLevels] = 'INFO'
LOGGING_LEVEL_DISPATCHER: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_INFSTRCT: Final[LoggingLevels] = 'INFO'
LOGGING_LEVEL_SOURCES: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_SINKS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_PRODSTATIONS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_JOBS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_OPERATIONS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_BUFFERS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_MONITORS: Final[LoggingLevels] = 'ERROR'
LOGGING_LEVEL_AGENTS: Final[LoggingLevels] = 'DEBUG'
LOGGING_LEVEL_CONDITIONS: Final[LoggingLevels] = 'DEBUG'
LOGGING_LEVEL_DB: Final[LoggingLevels] = 'DEBUG'

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
monitors = logging.getLogger('pyforcesim.monitors')
monitors.setLevel(LOGGING_LEVEL_MONITORS)
agents = logging.getLogger('pyforcesim.agents')
agents.setLevel(LOGGING_LEVEL_AGENTS)
conditions = logging.getLogger('pyforcesim.conditions')
conditions.setLevel(LOGGING_LEVEL_CONDITIONS)
databases = logging.getLogger('pyforcesim.databases')
databases.setLevel(LOGGING_LEVEL_DB)

jobs = logging.getLogger('pyforcesim.jobs')
jobs.setLevel(LOGGING_LEVEL_JOBS)
operations = logging.getLogger('pyforcesim.operations')
operations.setLevel(LOGGING_LEVEL_OPERATIONS)
