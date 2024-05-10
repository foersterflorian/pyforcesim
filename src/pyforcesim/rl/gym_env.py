import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pyforcesim.simulation.environment as sim
from pyforcesim.simulation import loads
from pyforcesim.datetime import (
    DTManager, 
    adjust_db_dates_local_tz,
)
from pyforcesim.rl import agents


def build_sim_env() -> tuple[
    sim.SimulationEnvironment,
    sim.InfrastructureManager,
    sim.Dispatcher,
    agents.AllocationAgent,
]:
    # !! REWORK, CURRENTLY ONLY FOR TESTING PURPOSES
    """Constructor to build simulation environment (layout)

    Returns
    -------
    tuple[ sim.SimulationEnvironment, sim.InfrastructureManager, sim.Dispatcher, agents.AllocationAgent, ]
        tuple out of Environment, InfrastructureManager, Dispatcher, Agent
    """
    # datetime manager
    dt_mgr = DTManager()
    starting_dt = dt_mgr.current_time_tz(cut_microseconds=True)
    # environment
    env = sim.SimulationEnvironment(name='base', time_unit='seconds', starting_datetime=dt)
    job_generator = loads.RandomJobGenerator(seed=2)
    infstruct_mgr = sim.InfrastructureManager(env=env)
    dispatcher = sim.Dispatcher(env=env, priority_rule='FIFO')
    
    # source
    area_source = sim.ProductionArea(env=env, custom_identifier=1000)
    group_source = sim.StationGroup(env=env, custom_identifier=1000)
    area_source.add_subsystem(group_source)
    proc_time = dt_mgr.timedelta_from_val(val=2., time_unit='hours')
    source = sim.Source(env=env, custom_identifier='source', proc_time=proc_time, 
                    random_generation=True, job_generator=job_generator, num_gen_jobs=4)
    group_source.add_subsystem(source)

    # sink
    area_sink = sim.ProductionArea(env=env, custom_identifier=2000)
    group_sink = sim.StationGroup(env=env, custom_identifier=2000)
    area_sink.add_subsystem(group_sink)
    sink = sim.Sink(env=env, custom_identifier='sink')
    group_sink.add_subsystem(sink)

    # processing stations
    # prod area 1
    area_prod = sim.ProductionArea(env=env, custom_identifier=1)
    group_prod = sim.StationGroup(env=env, custom_identifier=1)
    area_prod.add_subsystem(group_prod)
    group_prod2 = sim.StationGroup(env=env, custom_identifier=2)
    area_prod.add_subsystem(group_prod2)
    # prod area 2
    #area_prod2 = ProductionArea(env=env, custom_identifier=2)
    #group_prod3 = StationGroup(env=env, custom_identifier=3)
    #area_prod2.add_subsystem(group_prod3)
    #area_prod.add_subsystem(group_prod3)
    ## machines
    for machine in range(3):
        buffer = sim.Buffer(capacity=20, env=env, custom_identifier=(10+machine))
        if machine == 5:
            MachInst = sim.Machine(env=env, custom_identifier=machine, buffers=[buffer], setup_time=5.)
        else:
            MachInst = sim.Machine(env=env, custom_identifier=machine, buffers=[buffer])
            
        if machine == 0:
            testMachInst = MachInst
        
        if machine < 2:
            group_prod.add_subsystem(buffer)
            group_prod.add_subsystem(MachInst)
        elif machine >= 2:
            group_prod2.add_subsystem(buffer)
            group_prod2.add_subsystem(MachInst)
        else:
            pass
            #group_prod3.add_subsystem(buffer)
            #group_prod3.add_subsystem(MachInst)
        

    add_machine_to_bottleneck: bool = False
    if add_machine_to_bottleneck:
        buffer = sim.Buffer(capacity=20, env=env, custom_identifier=(10+machine+1))
        MachInst = sim.Machine(env=env, custom_identifier=machine+1, buffers=[buffer])
        group_prod3.add_subsystem(buffer)
        group_prod3.add_subsystem(MachInst)
        
    alloc_agent = agents.AllocationAgent(assoc_system=area_prod)

    # conditions
    duration_transient = dt_mgr.timedelta_from_val(val=2, time_unit='hours')
    trans_cond = sim.TransientCondition(env=env, duration_transient=duration_transient)
    agent_decision_cond = sim.TriggerAgentCondition(env=env)
    sim_dur = dt_parser.timedelta_from_val(val=2., time_unit='days')
    sim_end_date = dt_mgr.dt_with_tz_UTC(2024,3,23,12)
    job_gen_dur_cond = sim.JobGenDurationCondition(env=env, target_obj=source, sim_run_duration=sim_dur)
    
    return env, infstruct_mgr, dispatcher, alloc_agent



class JSSEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self) -> None:
        super().__init__()
        
        # build env
        (self.sim_env, self.infstruct_mgr, 
         self.dispatcher, self.agent) = build_sim_env()
        # action space for allocation agent is length of all associated 
        # infrastructure objects
        n_actions = len(self.agent.assoc_infstrct_objs)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # number of discrete actions depends on layout and infrastructure
        self.action_space = spaces.Discrete(n=n_actions)
        # Example for using image as input (channel-first; channel-last also works):
        # TODO change observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        
        self.terminated: bool = False

    def step(self, action):
        # process given action
        # step through sim_env till new decision should be made
        # calculate reward based on new observation
        
        ## ** action is provided as parameter, set action
        # ?? should still be checked? necessary?
        # should not be needed anymore, empty event list is checked below
        if self.env._event_list:
            print('Dispatching Signal', self.agent.dispatching_signal)
            self.agent.set_decision(action=action)
        else:
            print('Run ended!')
        
        # ** Run till next action is needed
        # execute with provided action till next decision should be made
        while not agent.dispatching_signal:
            
            # empty event list, simulation run ended
            if not self.env._event_list:
                self.terminated = True
                break
            
            self.env.step()
        
        # ** Calculate Reward
        # in agent class, not implemented yet
        # call from here

        # additional info
        truncated = {}
        info = {}
        
        # finalise simulation environment
        if self.terminated:
            self.sim_env.finalise()
        
        return observation, reward, self.terminated, truncated, info

    def reset(self, seed=None, options=None):
        # re-init simulation environment
        (self.sim_env, self.infstruct_mgr, 
         self.dispatcher, self.agent) = build_sim_env()
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
            if not self.env._event_list:
                self.terminated = True
                break
            self.env.step()
        
        # feature vector already built internally when dispatching signal is set
        observation = self.agent.feat_vec
        # ?? leave additional info empty?
        info = {}
        
        return observation, info

    def render(self):
        ...

    def close(self):
        ...