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
        )
        prod_sequence_PA = sequence_generator.retrieve(
            target_obj=source,
            uniform_lb=2,
            uniform_ub=4,
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

    observer1 = conditions.Observer(env=env, name='trigger_set')
    stop_prod_event = threading.Event()
    kwargs = dict(stop_production_event=stop_prod_event, prod_area=area_prod)
    env.register_observer(observer1, kwargs=kwargs)
    # observer2 = conditions.TestObserver(env=env, name='trigger_pull')
    # kwargs = dict(stop_production_event=stop_prod_event)
    # env.register_observer(observer2, kwargs=kwargs)

    return env, alloc_agent, seq_agent


# def test_agent_env() -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     # area_source.add_subsystem(group_source)
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # group_source.add_subsystem(source)
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))
#     # group_sink.add_subsystem(sink)

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     # area_prod.add_subsystem(group_prod)
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     conditions.TriggerAgentCondition(env=env, agent=alloc_agent)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     # machines
#     for machine in range(3):
#         if machine < 2:
#             target_group_prod = group_prod
#         else:
#             target_group_prod = group_prod2

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     return env, alloc_agent


# def standard_env_1_2_3_ConstIdeal(
#     sequencing: bool = False,
#     with_agent: bool = False,
#     validate: bool = False,
#     seed: int | None = None,
#     debug: bool = False,
# ) -> EnvAgentConstructorReturn:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     # env.dispatcher.alloc_rule = 'LOAD_TIME'
#     env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     # area_source.add_subsystem(group_source)
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     job_gen_limit: int | None = None
#     if debug:
#         job_gen_limit = 12

#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=job_gen_limit,
#     )
#     # group_source.add_subsystem(source)
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_buffer = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('buffer_group')
#     )

#     log_q: sim.LogicalQueue['Job'] | None = None
#     if sequencing:
#         log_q = cast(
#             sim.LogicalQueue['Job'],
#             sim.LogicalQueue(env=env, custom_identifier=CustomID('logQ_seq')),
#         )

#     buffer = sim.Buffer(
#         capacity=sim.MAX_LOGICAL_QUEUE_SIZE,
#         env=env,
#         supersystem=group_buffer,
#         custom_identifier=CustomID('buffer'),
#     )
#     # machines
#     for machine in range(3):
#         if machine < 2:
#             target_group_prod = group_prod
#         else:
#             target_group_prod = group_prod2

#         if not sequencing:
#             buffer = sim.Buffer(
#                 capacity=10_000,
#                 env=env,
#                 supersystem=target_group_prod,
#                 custom_identifier=CustomID(str(10 + machine).zfill(2)),
#             )

#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             logical_queue=log_q,
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     if not debug:
#         sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#         conditions.JobGenDurationCondition(
#             env=env, target_obj=source, sim_run_duration=sim_dur
#         )

#     alloc_agent: agents.AllocationAgent | None = None
#     seq_agent: agents.SequencingAgent | None = None

#     if validate:
#         alloc_agent = agents.ValidateSequencingAgent(assoc_system=area_prod)

#     if not sequencing and with_agent:
#         alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)
#     elif sequencing and with_agent:
#         assert log_q is not None, 'logical queue instance >>None<<'
#         seq_agent = agents.SequencingAgent(assoc_system=log_q)
#         conditions.TriggerAgentCondition(env=env, agent=seq_agent)

#     return env, alloc_agent, seq_agent


# def standard_env_1_3_7_VarIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     # env.dispatcher.alloc_rule = 'LOAD_TIME'
#     env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=1400,  # 1400
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     # machines
#     for machine in range(7):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         else:
#             target_group_prod = group_prod3

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.VariableSequenceSinglePA(
#         env=env,
#         seed=None,  # use env's default seed
#         prod_area_id=area_prod.system_id,
#     )
#     assert sequence_generator.seed == env.seed, 'seeds of sequence and env do not match'
#     logger.debug('Seed of env: %s, Seed of sequence: %s', env.seed, sequence_generator.seed)
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#         delta_percentage=0.35,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=14, time_unit=TimeUnitsTimedelta.DAYS)
#     # duration_transient = pyf_dt.timedelta_from_val(
#     #     val=1, time_unit=TimeUnitsTimedelta.SECONDS
#     # )
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=18, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_2_3_ConstIdeal_validate(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     # area_source.add_subsystem(group_source)
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # group_source.add_subsystem(source)
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))
#     # group_sink.add_subsystem(sink)

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     # area_prod.add_subsystem(group_prod)
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     # area_prod.add_subsystem(group_prod2)
#     # machines
#     for machine in range(3):
#         if machine < 2:
#             target_group_prod = group_prod
#         else:
#             target_group_prod = group_prod2

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=6, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.ValidateSequencingAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_3_7_ConstIdeal(
#     sequencing: bool = False,
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=21,  # None
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_buffer = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('buffer_group')
#     )

#     log_q = None
#     if sequencing:
#         log_q = sim.LogicalQueue(env=env, custom_identifier=CustomID('logQ_seq'))

#     buffer = sim.Buffer(
#         capacity=sim.MAX_LOGICAL_QUEUE_SIZE,
#         env=env,
#         supersystem=group_buffer,
#         custom_identifier=CustomID('buffer'),
#     )
#     # machines
#     for machine in range(7):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         else:
#             target_group_prod = group_prod3

#         if not sequencing:
#             buffer = sim.Buffer(
#                 capacity=20,
#                 env=env,
#                 supersystem=target_group_prod,
#                 custom_identifier=CustomID(str(10 + machine).zfill(2)),
#             )

#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             logical_queue=log_q,
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_3_7_ConstIdeal_validate(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     # machines
#     for machine in range(7):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         else:
#             target_group_prod = group_prod3

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.ValidateSequencingAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_3_7_VarIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     # env.dispatcher.alloc_rule = 'LOAD_TIME'
#     env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=1400,  # 1400
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     # machines
#     for machine in range(7):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         else:
#             target_group_prod = group_prod3

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.VariableSequenceSinglePA(
#         env=env,
#         seed=None,  # use env's default seed
#         prod_area_id=area_prod.system_id,
#     )
#     assert sequence_generator.seed == env.seed, 'seeds of sequence and env do not match'
#     logger.debug('Seed of env: %s, Seed of sequence: %s', env.seed, sequence_generator.seed)
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#         delta_percentage=0.35,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=14, time_unit=TimeUnitsTimedelta.DAYS)
#     # duration_transient = pyf_dt.timedelta_from_val(
#     #     val=1, time_unit=TimeUnitsTimedelta.SECONDS
#     # )
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=18, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_3_7_VarIdeal_validate(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     # env.dispatcher.alloc_rule = 'LOAD_TIME'
#     env.dispatcher.alloc_rule = 'LOAD_TIME_REMAINING'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=1400,  # 1400
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     # area_sink.add_subsystem(group_sink)
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     # machines
#     for machine in range(7):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         else:
#             target_group_prod = group_prod3

#         buffer = sim.Buffer(
#             capacity=20,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(10 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.VariableSequenceSinglePA(
#         env=env,
#         seed=None,  # use env's default seed
#         prod_area_id=area_prod.system_id,
#     )
#     # assert sequence_generator.seed == env.seed, 'seeds of sequence and env do not match'
#     logger.debug('Seed of env: %s, Seed of sequence: %s', env.seed, sequence_generator.seed)
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#         delta_percentage=0.35,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=28, time_unit=TimeUnitsTimedelta.HOURS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
#     # duration_transient = pyf_dt.timedelta_from_val(val=14, time_unit=TimeUnitsTimedelta.DAYS)
#     # duration_transient = pyf_dt.timedelta_from_val(
#     #     val=1, time_unit=TimeUnitsTimedelta.SECONDS
#     # )
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     # sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=12, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_dur = pyf_dt.timedelta_from_val(val=18, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.ValidateSequencingAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_5_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(5):
#         if machine < 1:
#             target_group_prod = group_prod
#         elif machine < 2:
#             target_group_prod = group_prod2
#         elif machine < 3:
#             target_group_prod = group_prod3
#         elif machine < 4:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_10_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(10):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 3:
#             target_group_prod = group_prod2
#         elif machine < 8:
#             target_group_prod = group_prod3
#         elif machine < 9:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_15_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(15):
#         if machine < 2:
#             target_group_prod = group_prod
#         elif machine < 5:
#             target_group_prod = group_prod2
#         elif machine < 7:
#             target_group_prod = group_prod3
#         elif machine < 12:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=15, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_20_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(20):
#         if machine < 4:
#             target_group_prod = group_prod
#         elif machine < 9:
#             target_group_prod = group_prod2
#         elif machine < 13:
#             target_group_prod = group_prod3
#         elif machine < 16:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=20, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_30_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(30):
#         if machine < 6:
#             target_group_prod = group_prod
#         elif machine < 15:
#             target_group_prod = group_prod2
#         elif machine < 22:
#             target_group_prod = group_prod3
#         elif machine < 26:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=30, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_50_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(50):
#         if machine < 9:
#             target_group_prod = group_prod
#         elif machine < 21:
#             target_group_prod = group_prod2
#         elif machine < 35:
#             target_group_prod = group_prod3
#         elif machine < 43:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=50, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent


# def standard_env_1_5_70_ConstIdeal(
#     with_agent: bool = False,
#     seed: int | None = None,
# ) -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
#     starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
#     env = sim.SimulationEnvironment(
#         name='base',
#         time_unit='seconds',
#         starting_datetime=starting_dt,
#         seed=seed,
#         debug_dashboard=False,
#     )
#     env.dispatcher.seq_rule = 'FIFO'
#     env.dispatcher.alloc_rule = 'LOAD_TIME'
#     # source
#     area_source = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1000'),
#         sim_get_prio=-20,
#         sim_put_prio=-30,
#     )
#     group_source = sim.StationGroup(
#         env=env, supersystem=area_source, custom_identifier=CustomID('1000')
#     )
#     order_time_source = pyf_dt.timedelta_from_val(val=1.0, time_unit=TimeUnitsTimedelta.HOURS)
#     source = sim.Source(
#         env=env,
#         supersystem=group_source,
#         custom_identifier=CustomID('source'),
#         proc_time=order_time_source,
#         job_generation_limit=None,
#     )
#     # sink
#     area_sink = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('2000'),
#         sim_get_prio=-22,
#         sim_put_prio=-32,
#     )
#     group_sink = sim.StationGroup(
#         env=env, supersystem=area_sink, custom_identifier=CustomID('2000')
#     )
#     _ = sim.Sink(env=env, supersystem=group_sink, custom_identifier=CustomID('sink'))

#     # processing stations
#     # prod area 1
#     area_prod = sim.ProductionArea(
#         env=env,
#         custom_identifier=CustomID('1'),
#         sim_get_prio=-21,
#         sim_put_prio=-31,
#     )
#     group_prod = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('1')
#     )
#     group_prod2 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('2')
#     )
#     group_prod3 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('3')
#     )
#     group_prod4 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('4')
#     )
#     group_prod5 = sim.StationGroup(
#         env=env, supersystem=area_prod, custom_identifier=CustomID('5')
#     )
#     # machines
#     for machine in range(70):
#         if machine < 15:
#             target_group_prod = group_prod
#         elif machine < 32:
#             target_group_prod = group_prod2
#         elif machine < 49:
#             target_group_prod = group_prod3
#         elif machine < 51:
#             target_group_prod = group_prod4
#         else:
#             target_group_prod = group_prod5

#         buffer = sim.Buffer(
#             capacity=1000,
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(100 + machine).zfill(2)),
#         )
#         _ = sim.Machine(
#             env=env,
#             supersystem=target_group_prod,
#             custom_identifier=CustomID(str(machine).zfill(2)),
#             buffers=[buffer],
#         )

#     sequence_generator = loads.ConstantSequenceSinglePA(
#         env=env, seed=100, prod_area_id=area_prod.system_id
#     )
#     prod_sequence_PA = sequence_generator.retrieve(
#         target_obj=source,
#     )
#     source.register_job_sequence(prod_sequence_PA)

#     # conditions
#     duration_transient = pyf_dt.timedelta_from_val(val=70, time_unit=TimeUnitsTimedelta.HOURS)
#     conditions.TransientCondition(env=env, duration_transient=duration_transient)
#     sim_dur = pyf_dt.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
#     # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
#     conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

#     alloc_agent = agents.AllocationAgent(assoc_system=area_prod)
#     if with_agent:
#         conditions.TriggerAgentCondition(env=env, agent=alloc_agent)

#     return env, alloc_agent
