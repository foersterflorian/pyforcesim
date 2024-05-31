import pytest
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


@pytest.fixture
def agent_policy():
    return AgentPolicy()


@pytest.fixture
def fifo_policy():
    return FIFOPolicy()


@pytest.fixture
def lifo_policy():
    return LIFOPolicy()


@pytest.fixture
def load_jobs_policy():
    return LoadJobsPolicy()


@pytest.fixture
def load_time_policy():
    return LoadTimePolicy()


@pytest.fixture
def lpt_policy():
    return LPTPolicy()


@pytest.fixture
def lst_policy():
    return LSTPolicy()


@pytest.fixture
def priority_policy():
    return PriorityPolicy()


@pytest.fixture
def random_policy():
    return RandomPolicy()


@pytest.fixture
def spt_policy():
    return SPTPolicy()


@pytest.fixture
def sst_policy():
    return SSTPolicy()


@pytest.fixture
def utilisation_policy():
    return UtilisationPolicy()


class Job:
    def __init__(self, current_proc_time, current_setup_time, prio):
        self.current_proc_time = current_proc_time
        self.current_setup_time = current_setup_time
        self.prio = prio


class ProcessingStation:
    def __init__(self, stat_monitor):
        self.stat_monitor = stat_monitor


class StatMonitor:
    def __init__(self, util, load_time, load_jobs):
        self.utilisation = util
        self.WIP_load_time = load_time
        self.WIP_load_num_jobs = load_jobs


@pytest.fixture(scope='module')
def numeric_queue():
    return tuple(i for i in range(1, 11))


@pytest.fixture(scope='module')
def job_queue():
    return tuple(Job(i, i, i) for i in range(1, 11))


@pytest.fixture(scope='module')
def processing_stations():
    return tuple(ProcessingStation(StatMonitor(i, i, i)) for i in range(1, 11))


def test_agent_policy(agent_policy, job_queue):
    with pytest.raises(NotImplementedError):
        agent_policy.apply(job_queue)


def test_random_policy(random_policy, numeric_queue):
    val = random_policy.apply(numeric_queue)
    expected = numeric_queue[1]
    assert val == expected


def test_fifo_policy(fifo_policy, job_queue):
    val = fifo_policy.apply(job_queue)
    expected = job_queue[0]
    assert val == expected


def test_lifo_policy(lifo_policy, job_queue):
    val = lifo_policy.apply(job_queue)
    expected = job_queue[-1]
    assert val == expected


def test_spt_policy(spt_policy, job_queue):
    val = spt_policy.apply(job_queue)
    expected = job_queue[0]
    assert val == expected


def test_lpt_policy(lpt_policy, job_queue):
    val = lpt_policy.apply(job_queue)
    expected = job_queue[-1]
    assert val == expected


def test_sst_policy(sst_policy, job_queue):
    val = sst_policy.apply(job_queue)
    expected = job_queue[0]
    assert val == expected


def test_lst_policy(lst_policy, job_queue):
    val = lst_policy.apply(job_queue)
    expected = job_queue[-1]
    assert val == expected


def test_priority_policy(priority_policy, job_queue):
    val = priority_policy.apply(job_queue)
    expected = job_queue[-1]
    assert val == expected


def test_utilisation_policy(processing_stations, utilisation_policy):
    val = utilisation_policy.apply(processing_stations)
    expected = processing_stations[0]
    assert val == expected


def test_load_time_policy(processing_stations, load_time_policy):
    val = load_time_policy.apply(processing_stations)
    expected = processing_stations[0]
    assert val == expected


def test_load_jobs_policy(processing_stations, load_jobs_policy):
    val = load_jobs_policy.apply(processing_stations)
    expected = processing_stations[0]
    assert val == expected
