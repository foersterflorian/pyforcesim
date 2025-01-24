from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar
from typing_extensions import override

import numpy as np

from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import (
    DISTRIBUTION_PARAMETERS,
    StatisticalDistributionsSupported,
    TimeUnitsTimedelta,
)
from pyforcesim.types import (
    StatDistributionInfo,
)

if TYPE_CHECKING:
    from numpy.random._generator import Generator as NPRandomGenerator

    from pyforcesim.simulation.environment import SimulationEnvironment
    from pyforcesim.types import (
        DistExpParameters,
        DistributionParametersSet,
        DistUniformParameters,
        StatDistributionInfo,
        Timedelta,
    )


T = TypeVar('T', bound='DistributionParametersSet')


def stat_info_uniform(
    uniform_lb: float,
    uniform_ub: float,
) -> StatDistributionInfo:
    if uniform_lb > uniform_ub:
        raise ValueError('Lower bound of distribution greater than upper bound.')

    expectation = (uniform_ub + uniform_lb) / 2
    std = (uniform_ub - uniform_lb) / (2 * np.sqrt(3))

    return StatDistributionInfo(mean=expectation, std=std)


def stat_info_exponential(
    scale: float,
) -> StatDistributionInfo:
    if scale <= 0:
        raise ValueError('Scale parameter must be greater than 0.')

    expectation = scale
    std = scale

    return StatDistributionInfo(mean=expectation, std=std)


class StatisticalDistribution(Generic[T], metaclass=ABCMeta):
    def __init__(
        self,
        env: SimulationEnvironment,
        params: T,
        dist_type: StatisticalDistributionsSupported,
        seed: int | None = None,
    ) -> None:
        self._env = env
        self._params = params
        # use env seed if applicable
        if seed is None and self.env.seed is not None:
            seed = self.env.seed
        self._seed = seed
        self._rnd_gen = np.random.default_rng(seed=seed)
        self._dist_type = dist_type
        self._stat_info: StatDistributionInfo | None = None

    @property
    def env(self) -> SimulationEnvironment:
        return self._env

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def rnd_gen(self) -> NPRandomGenerator:
        return self._rnd_gen

    @property
    def dist_type(self) -> StatisticalDistributionsSupported:
        return self._dist_type

    @property
    def stat_info(self) -> StatDistributionInfo:
        if self._stat_info is None:
            self._stat_info = self._calc_stat_info()
        return self._stat_info

    @property
    def params(self) -> T:
        return self._params

    def set_params(self, params: T) -> None:
        self._params = params
        self._stat_info = self._calc_stat_info()

    def sample_timedelta(
        self,
        time_unit: TimeUnitsTimedelta = TimeUnitsTimedelta.HOURS,
        round_to_minutes: bool = True,
    ) -> Timedelta:
        value = self.sample()
        td = pyf_dt.timedelta_from_val(value, time_unit)
        if round_to_minutes:
            td = pyf_dt.round_td_by_seconds(td, round_to_next_seconds=60)

        return td

    @abstractmethod
    def _calc_stat_info(self) -> StatDistributionInfo: ...

    @classmethod
    @abstractmethod
    def get_param_type(cls) -> type[DistributionParametersSet]: ...

    @abstractmethod
    def sample(self) -> float: ...


class Exponential(StatisticalDistribution['DistExpParameters']):
    def __init__(
        self,
        env: SimulationEnvironment,
        params: DistExpParameters,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            env=env,
            params=params,
            dist_type=StatisticalDistributionsSupported.EXPONENTIAL,
            seed=seed,
        )

    @override
    def _calc_stat_info(self) -> StatDistributionInfo:
        return stat_info_exponential(scale=self.params['scale'])

    @classmethod
    @override
    def get_param_type(cls) -> type[DistExpParameters]:
        return DISTRIBUTION_PARAMETERS.EXPONENTIAL

    @override
    def sample(self) -> float:
        return self.rnd_gen.exponential(scale=self.params['scale'])


class Uniform(StatisticalDistribution['DistUniformParameters']):
    def __init__(
        self,
        env: SimulationEnvironment,
        params: DistUniformParameters,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            env=env,
            params=params,
            dist_type=StatisticalDistributionsSupported.UNIFORM,
            seed=seed,
        )

    @override
    def _calc_stat_info(self) -> StatDistributionInfo:
        return stat_info_uniform(
            uniform_lb=self.params['lower_bound'],
            uniform_ub=self.params['upper_bound'],
        )

    @classmethod
    @override
    def get_param_type(cls) -> type[DistUniformParameters]:
        return DISTRIBUTION_PARAMETERS.UNIFORM

    @override
    def sample(self) -> float:
        return self.rnd_gen.uniform(
            low=self.params['lower_bound'],
            high=self.params['upper_bound'],
        )
