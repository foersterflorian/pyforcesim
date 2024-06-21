from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.simulation.policies import FIFOPolicy, LoadTimePolicy
from pyforcesim.types import CustomID


def test_base_env(env, starting_dt):
    assert env.name() == 'base'
    assert env.starting_datetime == starting_dt


def build_sim_env(env):
    # source
    area_source = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1000'),
        sim_get_prio=-20,
        sim_put_prio=-30,
    )
    group_source = sim.StationGroup(env=env, custom_identifier=CustomID('1000'))
    area_source.add_subsystem(group_source)
    order_time_source = pyf_dt.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    source = sim.Source(
        env=env,
        custom_identifier=CustomID('source'),
        proc_time=order_time_source,
        job_generation_limit=12,
    )
    group_source.add_subsystem(source)
    # sink
    area_sink = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('2000'),
        sim_get_prio=-22,
        sim_put_prio=-32,
    )
    group_sink = sim.StationGroup(env=env, custom_identifier=CustomID('2000'))
    area_sink.add_subsystem(group_sink)
    sink = sim.Sink(env=env, custom_identifier=CustomID('sink'))
    group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(
        env=env,
        custom_identifier=CustomID('1'),
        sim_get_prio=-21,
        sim_put_prio=-31,
    )
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

    # job generation
    sequence_generator = loads.ConstantSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.retrieve(target_obj=source)
    source.register_job_sequence(prod_sequence_PA)

    # conditions
    duration_transient = pyf_dt.timedelta_from_val(val=2, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    # agent_decision_cond = conditions.TriggerAgentCondition(env=env)
    # sim_dur = pyf_dt.timedelta_from_val(val=1, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = pyf_dt.dt_with_tz_UTC(2024, 3, 23, 12)
    # conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    return env, alloc_agent, MachInst


def export_results(env: sim.SimulationEnvironment, machine: sim.Machine):
    _ = env.dispatcher.draw_gantt_chart(dates_to_local_tz=False, save_html=True)
    _ = machine.stat_monitor.draw_state_chart(save_html=True, pie_chart=False)
    _ = machine.stat_monitor.draw_state_chart(save_html=True, pie_chart=True)


def test_build_env(env):
    env, agent, machine = build_sim_env(env)
    assert isinstance(agent, agents.AllocationAgent)
    assert env.check_integrity() is None
    assert env.dispatcher.seq_rule == 'FIFO'
    assert env.dispatcher.alloc_rule == 'LOAD_TIME'
    assert isinstance(env.dispatcher.seq_policy, FIFOPolicy)
    assert isinstance(env.dispatcher.alloc_policy, LoadTimePolicy)
    assert env.initialise() is None
    assert env.run() is None
    assert env.finalise() is None
    # assert env.dispatcher.cycle_time == datetime.timedelta(seconds=57600)
    export_results(env, machine)
