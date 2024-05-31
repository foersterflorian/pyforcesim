import pytest
from pyforcesim.datetime import DTManager


@pytest.fixture(scope='session')
def dt_manager():
    return DTManager()
