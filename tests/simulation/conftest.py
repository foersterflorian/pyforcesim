import pytest
from pyforcesim.datetime import DTManager
from pyforcesim.simulation import environment as sim


@pytest.fixture(scope='module')
def env(starting_dt):
    env = sim.SimulationEnvironment(
        name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
    )
    return env
