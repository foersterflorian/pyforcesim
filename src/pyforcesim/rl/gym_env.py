from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.datetime import DTManager
from pyforcesim.rl import agents
from pyforcesim.simulation import conditions, loads
from pyforcesim.simulation import environment as sim
from pyforcesim.simulation.policies import FIFOPolicy, LoadTimePolicy
from pyforcesim.types import CustomID


def build_sim_env() -> tuple[sim.SimulationEnvironment, agents.AllocationAgent]:
    dt_manager = DTManager()
    starting_dt = dt_manager.dt_with_tz_UTC(2024, 3, 28, 0)
    env = sim.SimulationEnvironment(
        name='base', time_unit='seconds', starting_datetime=starting_dt, debug_dashboard=False
    )
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

    # source
    area_source = sim.ProductionArea(env=env, custom_identifier=CustomID('1000'))
    group_source = sim.StationGroup(env=env, custom_identifier=CustomID('1000'))
    area_source.add_subsystem(group_source)
    proc_time = dt_manager.timedelta_from_val(val=2.0, time_unit=TimeUnitsTimedelta.HOURS)
    sequence_generator = loads.ProductionSequenceSinglePA(
        env=env, seed=100, prod_area_id=area_prod.system_id
    )
    prod_sequence_PA = sequence_generator.constant_sequence(order_time_source=proc_time)
    source = sim.Source(
        env=env,
        custom_identifier=CustomID('source'),
        proc_time=proc_time,
        job_sequence=prod_sequence_PA,
        num_gen_jobs=None,
    )
    group_source.add_subsystem(source)

    # conditions
    duration_transient = dt_manager.timedelta_from_val(
        val=12, time_unit=TimeUnitsTimedelta.HOURS
    )
    conditions.TransientCondition(env=env, duration_transient=duration_transient)
    # agent_decision_cond = conditions.TriggerAgentCondition(env=env)
    sim_dur = dt_manager.timedelta_from_val(val=3, time_unit=TimeUnitsTimedelta.WEEKS)
    # sim_end_date = dt_manager.dt_with_tz_UTC(2024, 3, 23, 12)
    conditions.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)

    return env, alloc_agent


class JSSEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # build env
        self.sim_env, self.agent = build_sim_env()
        # action space for allocation agent is length of all associated
        # infrastructure objects
        n_actions = len(self.agent.assoc_proc_stations)
        self.action_space = gym.spaces.Discrete(n=n_actions, seed=seed)
        # Example for using image as input (channel-first; channel-last also works):
        # TODO change observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(), dtype=np.float32)

        self.terminated: bool = False

    def step(
        self,
        action: int,
    ) -> tuple[npt.NDArray[np.float32], float, bool, dict, dict]:
        # process given action
        # step through sim_env till new decision should be made
        # calculate reward based on new observation

        ## ** action is provided as parameter, set action
        # ?? should still be checked? necessary?
        # should not be needed anymore, empty event list is checked below
        self.agent.set_decision(action=action)

        # ** Run till next action is needed
        # execute with provided action till next decision should be made
        while not self.agent.dispatching_signal:
            # empty event list, simulation run ended
            if not self.sim_env._event_list:
                self.terminated = True
                break

            self.sim_env.step()

        # ** Calculate Reward
        # in agent class, not implemented yet
        # call from here
        reward = self.agent.calc_reward()
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in step!')

        # additional info
        truncated = {}
        info = {}

        # finalise simulation environment
        if self.terminated:
            self.sim_env.finalise()

        return observation, reward, self.terminated, truncated, info

    def reset(
        self,
        seed: int = 42,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        # re-init simulation environment
        self.sim_env, self.agent = build_sim_env()
        # evaluate if all needed components are registered
        self.sim_env.check_integrity()
        # initialise simulation environment
        self.sim_env.initialise()

        # run till first decision should be made
        # transient condition implemented --> triggers a point in time
        # at which agent makes decisions

        # ** Run till settling process is finished
        while not self.agent.dispatching_signal:
            # empty event list, simulation run ended
            # theoretically should never be triggered unless transient condition
            # is met later than configured simulation time
            if not self.sim_env._event_list:
                self.terminated = True
                break
            self.sim_env.step()
        # feature vector already built internally when dispatching signal is set
        observation = self.agent.feat_vec
        if observation is None:
            raise ValueError('No Observation in reset!')
        info = {}

        return observation, info

    def render(self): ...

    def close(self): ...
