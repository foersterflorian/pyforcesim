from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.datetime import DTManager
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.types import CustomID


def build_sim_env() -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    dt_manager = DTManager()
    starting_dt = dt_manager.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
    )
    env.dispatcher.seq_rule = 'FIFO'
    env.dispatcher.alloc_rule = 'LOAD_TIME'
    # source
    area_source = sim.ProductionArea(env=env, custom_identifier=CustomID('1000'))
    group_source = sim.StationGroup(env=env, custom_identifier=CustomID('1000'))
    area_source.add_subsystem(group_source)
    order_time_source = dt_manager.timedelta_from_val(
        val=2.0, time_unit=TimeUnitsTimedelta.HOURS
    )
    source = sim.Source(
        env=env,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=None,
    )
    group_source.add_subsystem(source)
    # sink
    area_sink = sim.ProductionArea(env=env, custom_identifier=CustomID('2000'))
    group_sink = sim.StationGroup(env=env, custom_identifier=CustomID('2000'))
    area_sink.add_subsystem(group_sink)
    sink = sim.Sink(env=env, custom_identifier=CustomID('sink'))
    group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(env=env, custom_identifier=CustomID('1'))
    group_prod = sim.StationGroup(env=env, custom_identifier=CustomID('1'))
    area_prod.add_subsystem(group_prod)
    group_prod2 = sim.StationGroup(env=env, custom_identifier=CustomID('2'))
    area_prod.add_subsystem(group_prod2)
    # machines
    for machine in range(3):
        buffer = sim.Buffer(
            capacity=20, env=env, custom_identifier=CustomID(str(10 + machine))
        )
        MachInst = sim.Machine(
            env=env, custom_identifier=CustomID(str(machine)), buffers=[buffer]
        )

        if machine < 2:
            group_prod.add_subsystem(buffer)
            group_prod.add_subsystem(MachInst)
        else:
            group_prod2.add_subsystem(buffer)
            group_prod2.add_subsystem(MachInst)

    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)

    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(
        target_obj=source,
    )
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = dt_manager.timedelta_from_val(
        val=2, time_unit=TimeUnitsTimedelta.HOURS
    )
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    conditions.TriggerAgentCondition(env=env)
    sim_dur = dt_manager.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = dt_manager.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    return env, alloc_agent
