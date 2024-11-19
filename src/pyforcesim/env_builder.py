import threading
from typing import TYPE_CHECKING, cast

import numpy as np

import pyforcesim.datetime as pyf_dt
from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.loggers import env_builder as logger
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.types import AgentType, CustomID, EnvAgentConstructorReturn

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import Job


def standard_env_single_area(
    sequencing: bool = False,
    with_agent: bool = False,
    validate: bool = False,
    seed: int | None = None,
    num_station_groups: int = 2,
    num_machines: int = 3,
    variable_source_sequence: bool = False,
    debug: bool = False,
    seed_layout: int | None = None,
    factor_WIP: float = 0,
) -> EnvAgentConstructorReturn:
    layout_rng = np.random.default_rng(seed=seed_layout)

    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    # env.dispatcher.alloc_rule = 'LOAD_TIME'
    env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
    # source
    area_source = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1000'),
        sim_get_prio=-20,
        sim_put_prio=-30,
    )
    group_source = sim.StationGroup(
        env=env, supersystem=area_source, custom_identifier=CustomID('1000')
    )
    # area_source.add_subsystem(group_source)
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    job_gen_limit: int | None = None
    if debug:
        job_gen_limit = 12

    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=job_gen_limit,
    )
    # group_source.add_subsystem(source)
    # sink
    area_sink = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('2000'),
        sim_get_prio=-22,
        sim_put_prio=-32,
    )
    group_sink = sim.StationGroup(
        env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
    )
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('prod_area_1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )

    stat_groups: list[sim.StationGroup] = []
    for i in range(num_station_groups):
        group_prod = sim.StationGroup(
            env=env, supersystem=area_prod, custom_identifier=CustomID(f'stat_group_{i}')
        )
        stat_groups.append(group_prod)

    log_q: sim.LogicalQueue['Job'] | None = None
    if sequencing:
        log_q = cast(
            sim.LogicalQueue['Job'],
            sim.LogicalQueue(env=env, custom_identifier=CustomID('logQ_seq')),
        )

    group_buffer = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('buffer_group')
    )
    buffer = sim.Buffer(
        capacity=sim.MAX_LOGICAL_QUEUE_SIZE,
        env=env,
        supersystem=group_buffer,
        custom_identifier=CustomID('buffer'),
    )
    # machines
    stat_group_indices = layout_rng.integers(0, num_station_groups, size=(num_machines,))
    stat_group_indices = np.sort(stat_group_indices)
    for machine in range(num_machines):
        # idx = layout_rng.integers(0, num_stat_groups)
        idx = stat_group_indices[machine]
        target_stat_group = stat_groups[idx]

        if not sequencing:
            buffer = sim.Buffer(
                capacity=10_000,
                env=env,
                supersystem=target_stat_group,
                custom_identifier=CustomID(str(10 + machine).zfill(2)),
            )

        _ = sim.Machine(
            env=env,
            supersystem=target_stat_group,
            custom_identifier=CustomID(str(machine).zfill(2)),
            logical_queue=log_q,
            buffers=[buffer],
        )

    sequence_generator: loads.SequenceSinglePA
    if variable_source_sequence:
        # sequence_generator = loads.VariableSequenceSinglePA(
        #     env=env,
        #     seed=None,  # use env's default seed
        #     prod_area_id=area_prod.system_id,
        # )
        # prod_sequence_PA = sequence_generator.retrieve(
        #     target_obj=source,
        #     delta_percentage=0.35,
        # )
        sequence_generator = loads.WIPSequenceSinglePA(
            env=env,
            seed=None,  # use env's default seed
            prod_area_id=area_prod.system_id,
            uniform_lb=2,
            uniform_ub=4,
        )
        prod_sequence_PA = sequence_generator.retrieve(
            target_obj=source,
            factor_WIP=factor_WIP,
        )
    else:
        sequence_generator = loads.ConstantSequenceSinglePA(
            env=env,
            seed=None,  # use env's default seed
            prod_area_id=area_prod.system_id,
        )
        prod_sequence_PA = sequence_generator.retrieve(
            target_obj=source,
        )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=8, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    if not debug:
        sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
        conditions.JobGenDurationCondition(
            env=env, target_obj=source, sim_run_duration=sim_dur
        )

    alloc_agent: agents.AllocationAgent | None = None
    seq_agent: agents.SequencingAgent | None = None
    if not sequencing and with_agent and not validate:
        alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    elif not sequencing and with_agent and validate:
        alloc_agent = agents.ValidateAllocationAgent(assoc_system=area_prod)
    elif sequencing and with_agent and not validate:
        assert log_q is not None, 'logical queue instance >>None<<'
        seq_agent = agents.SequencingAgent(assoc_system=log_q)
    elif sequencing and with_agent and validate:
        assert log_q is not None, 'logical queue instance >>None<<'
        seq_agent = agents.ValidateSequencingAgent(assoc_system=log_q)

    if with_agent:
        agent: AgentType
        if alloc_agent is not None:
            agent = alloc_agent
        elif seq_agent is not None:
            agent = seq_agent
        else:
            raise ValueError('Agent not provided for Env with agent decision')

        conditions.TriggerAgentCondition(env=env, agent=agent)

    stop_prod_event = threading.Event()
    WIP_limit = pyf_dt.timedelta_from_val(40, TimeUnitsTimedelta.HOURS)
    controller_interval = pyf_dt.timedelta_from_val(30, TimeUnitsTimedelta.MINUTES)
    WIP_observer = conditions.WIPSourceController(
        env,
        'WIP-Observer',
        stop_prod_event,
        sim_interval=controller_interval,
        prod_area=area_prod,
        target_sources=(source,),
        WIP_limit=WIP_limit,
    )
    env.register_observer(WIP_observer)  # TODO: currently pointless

    return env, alloc_agent, seq_agent
