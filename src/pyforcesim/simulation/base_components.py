from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import salabim

from pyforcesim.types import Infinite

if TYPE_CHECKING:
    from pyforcesim.simulation.environment import SimulationEnvironment

# set Salabim to yield mode (using yield is mandatory)
salabim.yieldless(False)


class SimulationComponent(salabim.Component):
    """thin wrapper for Salabim components to add them as component
    to simulation entities"""

    def __init__(
        self,
        env: SimulationEnvironment,
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


class CustomSalabimStore(salabim.Store):
    """class to override the not defined setup method for Salabim store components
    which allows to successfully pass keyword arguments to the constructor of the
    store component and finally to the associated queue class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs) -> None:
        pass


class StorageComponent(SimulationComponent):
    """thin wrapper for Salabim store components to add them as component
    to simulation entities"""

    def __init__(
        self,
        env: SimulationEnvironment,
        name: str,
        capacity: int | Infinite,
        pre_process: Callable[..., Any],
        sim_logic: Callable[..., Generator[Any, Any, Any]],
        post_process: Callable[..., Any],
    ):
        super().__init__(
            env=env,
            name=name,
            pre_process=pre_process,
            sim_logic=sim_logic,
            post_process=post_process,
        )
        storage_name = f'{name}_storage'
        self._store = CustomSalabimStore(
            env=env,
            name=storage_name,
            capacity=capacity,  # type: ignore Salabim wrong type hint
            monitor=False,
        )
        self._store_name = self._store.name()

    @property
    def store(self) -> salabim.Store:
        return self._store

    @property
    def store_name(self) -> str:
        return self._store_name
