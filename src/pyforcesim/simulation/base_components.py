from typing import TYPE_CHECKING, Any
from collections.abc import Generator, Callable

import salabim


if TYPE_CHECKING:
    from pyforcesim.simulation.environment import SimulationEnvironment

# set Salabim to yield mode (using yield is mandatory)
salabim.yieldless(False)

class SimulationComponent(salabim.Component):
    """thin wrapper for Salabim components to add them as component
    to simulation entities"""
    
    def __init__(
        self,
        env: 'SimulationEnvironment',
        name: str,
        pre_process: Callable[..., Any],
        sim_logic: Callable[..., Generator[Any, Any, Any]],
        post_process: Callable[..., Any],
    ):
        self.pyfsim_env = env
        self.pre_process = pre_process
        self.sim_logic = sim_logic
        self.post_process = post_process
        
        super().__init__(env=env, process='run', suppress_trace=True, name=name)
    
    def run(self) -> Generator[Any, Any, None]:
        """main logic loop for all resources in the simulation environment"""
        #logger_infstrct.debug(f"----> Process logic of {self}")
        # pre control logic
        ret = self.pre_process()
        # main control logic
        if ret is not None:
            ret = yield from self.sim_logic(*ret)
        else:
            ret = yield from self.sim_logic()
        # post control logic
        if ret is not None:
            ret = self.post_process(*ret)
        else:
            ret = self.post_process()

