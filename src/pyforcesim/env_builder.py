import pyforcesim.datetime as pyf_dt
from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.loggers import env_builder as logger
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.types import CustomID


def test_agent_env() -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))
    # group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    # area_prod.add_subsystem(group_prod)
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    conditions.TriggerAgentCondition(env=env, agent=alloc_agent)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    # machines
    for machine in range(3):
        if machine < 2:
            target_group_prod = group_prod
        else:
            target_group_prod = group_prod2

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    return env, alloc_agent


def standard_env_1_2_3_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))
    # group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    # area_prod.add_subsystem(group_prod)
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    # area_prod.add_subsystem(group_prod2)
    # machines
    for machine in range(3):
        if machine < 2:
            target_group_prod = group_prod
        else:
            target_group_prod = group_prod2

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_2_3_ConstIdeal_validate(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))
    # group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    # area_prod.add_subsystem(group_prod)
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    # area_prod.add_subsystem(group_prod2)
    # machines
    for machine in range(3):
        if machine < 2:
            target_group_prod = group_prod
        else:
            target_group_prod = group_prod2

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.ValidateAllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_3_7_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    # machines
    for machine in range(7):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 5:
            target_group_prod = group_prod2
        else:
            target_group_prod = group_prod3

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_3_7_ConstIdeal_validate(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    # machines
    for machine in range(7):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 5:
            target_group_prod = group_prod2
        else:
            target_group_prod = group_prod3

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.ValidateAllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_3_7_VarIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
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
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=1400,
    )
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    # machines
    for machine in range(7):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 5:
            target_group_prod = group_prod2
        else:
            target_group_prod = group_prod3

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.VariableSequenceSinglePA(
        env=env,
        seed=None,  # use env's default seed
        prod_area_id=area_prod.system_id,
    )
    assert sequence_generator.seed == env.seed, 'seeds of sequence and env do not match'
    logger.debug('Seed of env: %s, Seed of sequence: %s', env.seed, sequence_generator.seed)
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
        delta_percentage=0.35,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    # duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
    duration_transient = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
    # duration_transient = pyf_dt.timedelta_from_val(val=14, time_unit=TimeUnitsTimedelta.DAYS)
    # duration_transient = pyf_dt.timedelta_from_val(
    #     val=1, time_unit=TimeUnitsTimedelta.SECONDS
    # )
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
    sim_dur = pyf_dt.timedelta_from_val(val=18, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_3_7_VarIdeal_validate(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
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
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=1400,
    )
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
    # area_sink.add_subsystem(group_sink)
    _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    # machines
    for machine in range(7):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 5:
            target_group_prod = group_prod2
        else:
            target_group_prod = group_prod3

        buffer = sim.Buffer(
            capacity=20,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(10 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.VariableSequenceSinglePA(
        env=env,
        seed=None,  # use env's default seed
        prod_area_id=area_prod.system_id,
    )
    # assert sequence_generator.seed == env.seed, 'seeds of sequence and env do not match'
    logger.debug('Seed of env: %s, Seed of sequence: %s', env.seed, sequence_generator.seed)
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
        delta_percentage=0.35,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    # duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
    duration_transient = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
    # duration_transient = pyf_dt.timedelta_from_val(val=14, time_unit=TimeUnitsTimedelta.DAYS)
    # duration_transient = pyf_dt.timedelta_from_val(
    #     val=1, time_unit=TimeUnitsTimedelta.SECONDS
    # )
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
    sim_dur = pyf_dt.timedelta_from_val(val=18, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.ValidateAllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_5_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(5):
        if machine < 1:
            target_group_prod = group_prod
        elif machine < 2:
            target_group_prod = group_prod2
        elif machine < 3:
            target_group_prod = group_prod3
        elif machine < 4:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_10_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(10):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 3:
            target_group_prod = group_prod2
        elif machine < 8:
            target_group_prod = group_prod3
        elif machine < 9:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_15_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(15):
        if machine < 2:
            target_group_prod = group_prod
        elif machine < 5:
            target_group_prod = group_prod2
        elif machine < 7:
            target_group_prod = group_prod3
        elif machine < 12:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_20_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(20):
        if machine < 4:
            target_group_prod = group_prod
        elif machine < 9:
            target_group_prod = group_prod2
        elif machine < 13:
            target_group_prod = group_prod3
        elif machine < 16:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=20, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_30_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(30):
        if machine < 6:
            target_group_prod = group_prod
        elif machine < 15:
            target_group_prod = group_prod2
        elif machine < 22:
            target_group_prod = group_prod3
        elif machine < 26:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=30, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_50_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(50):
        if machine < 9:
            target_group_prod = group_prod
        elif machine < 21:
            target_group_prod = group_prod2
        elif machine < 35:
            target_group_prod = group_prod3
        elif machine < 43:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=50, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent


def standard_env_1_5_70_ConstIdeal(
    with_agent: bool = False,
    seed: int | None = None,
) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
        debug_dashboard=False,
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
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
    order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        supersystem=group_source,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
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
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
    group_prod = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('1')
    )
    group_prod2 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('2')
    )
    group_prod3 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('3')
    )
    group_prod4 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('4')
    )
    group_prod5 = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('5')
    )
    # machines
    for machine in range(70):
        if machine < 15:
            target_group_prod = group_prod
        elif machine < 32:
            target_group_prod = group_prod2
        elif machine < 49:
            target_group_prod = group_prod3
        elif machine < 51:
            target_group_prod = group_prod4
        else:
            target_group_prod = group_prod5

        buffer = sim.Buffer(
            capacity=1000,
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(100 + machine).zfill(2)),
        )
        _ = sim.Machine(
            env=env,
            supersystem=target_group_prod,
            custom_identifier=CustomID(str(machine).zfill(2)),
            buffers=[buffer],
        )

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=70, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
    if with_agent:
        conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

    return env, alloc_agent
