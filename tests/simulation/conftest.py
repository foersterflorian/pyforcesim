import pytest
from pyforcesim.datetime import DTManager
from pyforcesim.simulation import environment as sim


@pytest.fixture(scope='module')
def env(starting_dt):
    env = sim.SimulationEnvironment(
        name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
    return env


@pytest.fixture(scope='session')
def starting_dt(dt_manager):
    starting_dt = dt_manager.dt_with_tz_UTC(2024, 3, 28, 0)
    return starting_dt
