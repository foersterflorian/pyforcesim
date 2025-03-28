from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

import pyforcesim.datetime as pyf_dt
from pyforcesim.constants import (
    MAX_LOGICAL_QUEUE_SIZE,
    WIP_LEVELS_FULL_CYCLES,
    TimeUnitsTimedelta,
)
from pyforcesim.loggers import env_builder as logger
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, distributions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.types import CustomID

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import Job
    from pyforcesim.types import AgentType, EnvAgentConstructorReturn, Timedelta


def calc_setter_interval_by_num_iterations(
    num_cycles: int,
    sim_duration: Timedelta,
    num_levels: int,
) -> Timedelta:
    duration_per_level = sim_duration / (num_cycles * num_levels)
    return pyf_dt.round_td_by_seconds(duration_per_level, round_to_next_seconds=60)


def calc_WIP_relative_targets(
    WIP_rel_min: float,
    WIP_rel_max: float,
    num_levels: int,
) -> tuple[float, tuple[float, ...]]:
    if num_levels % 2 == 0:
        raise ValueError('Number of WIP levels must be odd')

    WIP_relative_targets: tuple[float, ...] = tuple(
        np.linspace(WIP_rel_min, WIP_rel_max, num=num_levels, dtype=np.float64).tolist()
    )
    mean = (WIP_rel_min + WIP_rel_max) / 2

    return mean, WIP_relative_targets


def standard_env_single_area(
    sequencing: bool = False,
    with_agent: bool = False,
    validate: bool = False,
    seed: int | None = None,
    sim_dur_weeks: float = 26,
    num_station_groups: int = 2,
    num_machines: int = 3,
    variable_source_sequence: bool = False,
    debug: bool = False,
    seed_layout: int | None = None,
    factor_WIP: float | None = None,  # default: overload condition
    WIP_relative_target: Sequence[float] = (1.5,),
    WIP_level_cycles: int = WIP_LEVELS_FULL_CYCLES,
    WIP_relative_planned: float = 1.5,  # util about 95 % with alpha = 7
    alpha: float = 7,
    buffer_size: int = MAX_LOGICAL_QUEUE_SIZE,
    job_pool_size_min: int = 1,
    job_pool_size_max: int = 1,
    dispatcher_seq_rule: str = 'FIFO',
    dispatcher_alloc_rule: str = 'LOAD_TIME_REMAINING',
) -> EnvAgentConstructorReturn:
    """constructor function for simulation environment of a single production area

    Parameters
    ----------
    sequencing : bool, optional
        decision between sequencing and validation, by default False
    with_agent : bool, optional
        implement agent or not, by default False
    validate : bool, optional
        use validation agent (set of rules) instead of RL-agent, by default False
    seed : int | None, optional
        RNG seed, by default None
    num_station_groups : int, optional
        number of station groups within environment, by default 2
    num_machines : int, optional
        number of stations within environment, by default 3
    variable_source_sequence : bool, optional
        use a variable sequence of jobs, by default False
    debug : bool, optional
        debug environment with short simulation duration, by default False
    seed_layout : int | None, optional
        seed for RNG controlling the random assignment of stations to station groups,
        by default None
    factor_WIP : float, optional
        defines the interval in which jobs arrive, value of 1 equals equilibrium:
        inflow = outflow, by default 3
    WIP_relative_target: Sequence[float], optional
        defines the WIP levels which should be used for WIP control, the system iterates
        through each entry repeatedly, by default (1.5,)
    WIP_relative_planned : float, optional
        planned WIP based on factor of ideal WIP calculated by system properties, defines how
        planned ending dates are calculated (assumes systematic planning approach),
        by default 1.5
    alpha: float, optional
        defines the curvature of the underlying logistics operation curve, by default 7
    buffer_size : int, optional
        buffer size of the shared buffer if sequencing is chosen,
        by default MAX_LOGICAL_QUEUE_SIZE

    Returns
    -------
    EnvAgentConstructorReturn
        simulation environment with corresponding agents if appropriate

    Raises
    ------
    ValueError
        if agent decision is desired and no agent was provided (should never occur)
    """
    layout_rng = np.random.default_rng(seed=seed_layout)

    starting_dt = pyf_dt.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base',
        time_unit='seconds',
        starting_datetime=starting_dt,
        seed=seed,
    )
    env.dispatcher.seq_rule = dispatcher_seq_rule
    env.dispatcher.alloc_rule = dispatcher_alloc_rule
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
        job_gen_limit = 24

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
            sim.LogicalQueue(
                env=env, custom_identifier=CustomID('logQ_seq'), size=buffer_size
            ),
        )

    group_buffer = sim.StationGroup(
        env=env, supersystem=area_prod, custom_identifier=CustomID('buffer_group')
    )
    buffer = sim.Buffer(
        capacity=buffer_size,
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
                capacity=100,
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

    # ** distributions and sequences
    DistParams = distributions.Uniform.get_param_type()
    dist_params = DistParams(lower_bound=2, upper_bound=4)
    dist_order = distributions.Uniform(env, dist_params)
    DistParams = distributions.Exponential.get_param_type()
    dist_params = DistParams(scale=1)  # set by WIPLimitSetter
    dist_arrival = distributions.Exponential(env, dist_params)

    sequence_generator: loads.SequenceSinglePA
    if variable_source_sequence:
        # ** using WIP base target to calculate lead time based on
        # ** a fixed planned WIP level
        sequence_generator = loads.WIPSequenceSinglePA(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            seed=None,  # use env's default seed
            prod_area_id=area_prod.system_id,
            WIP_relative_planned=WIP_relative_planned,
            WIP_alpha=alpha,
        )
        if factor_WIP is None:
            factor_WIP = WIP_relative_target[0]

        prod_sequence_PA = sequence_generator.retrieve(
            WIP_relative=factor_WIP,
            job_pool_size_min=job_pool_size_min,
            job_pool_size_max=job_pool_size_max,
            shuffle_job_pool=True,
            random_due_date_diff=False,
        )
    else:
        sequence_generator = loads.ConstantSequenceSinglePA(
            env=env,
            dist_order=dist_order,
            dist_arrival=dist_arrival,
            source=source,
            seed=None,  # use env's default seed
            prod_area_id=area_prod.system_id,
        )
        prod_sequence_PA = sequence_generator.retrieve()

    source.register_job_sequence(prod_sequence_PA)

    # conditions
    # ** simulation duration and transient condition
    duration_transient = pyf_dt.timedelta_from_val(val=8, time_unit=TimeUnitsTimedelta.HOURS)
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    sim_dur: Timedelta | None = None
    if not debug:
        sim_dur = pyf_dt.timedelta_from_val(
            val=sim_dur_weeks, time_unit=TimeUnitsTimedelta.WEEKS
        )
        conditions.JobGenDurationCondition(
            env=env, target_obj=source, sim_run_duration=sim_dur
        )

    # ** agents
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

    # ** WIP control
    WIP_supervisor = conditions.WIPSourceSupervisor(
        env,
        'WIP-Supervisor',
        prod_area=area_prod,
        supervised_sources=(source,),
        WIP_relative=WIP_relative_target[0],
        stat_info_orders=dist_order.stat_info,
        WIP_time=None,
    )

    num_relative_targets = len(WIP_relative_target)
    if num_relative_targets > 1 and sim_dur is not None:
        # WIP_setter_interval = pyf_dt.timedelta_from_val(2, TimeUnitsTimedelta.WEEKS)
        WIP_setter_interval = calc_setter_interval_by_num_iterations(
            num_cycles=WIP_level_cycles,
            sim_duration=sim_dur,
            num_levels=num_relative_targets,
        )
        _ = conditions.WIPSourceSupervisorLimitSetter(
            env,
            'WIP-Limit-Setter',
            WIP_setter_interval,
            WIP_source_supervisor=WIP_supervisor,
            WIP_relatives=WIP_relative_target,
            WIP_times=None,
        )
        logger.info(
            (
                '[ENV-BUILDER] Successfully added WIPLimitSetter with %d interval(s) '
                'and setter interval >>%s<<.'
            ),
            num_relative_targets,
            WIP_setter_interval,
        )

    logger.info('[ENV-BUILDER] Successfully created new simulation environment.')

    return env, alloc_agent, seq_agent
