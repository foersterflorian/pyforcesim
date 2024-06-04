from __future__ import annotations

from collections.abc import Hashable, Iterator
from enum import Enum, IntEnum, StrEnum
from typing import Any, Type, overload, Literal

from pyforcesim.types import FlattableObject


def flatten(
    obj: FlattableObject,
) -> Iterator[Any]:
    """flattens an arbitrarily nested list or tuple
    https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists

    Parameters
    ----------
    obj : FlattableObject
        arbitrarily nested list, tuple, set

    Yields
    ------
    Iterator[Any]
        elements of the non-nested list, tuple, set
    """
    for x in obj:
        # only flatten lists and tuples
        if isinstance(x, (list, tuple, set)):
            yield from flatten(x)
        else:
            yield x


def enum_str_values_as_frzset(
    enum: Type[StrEnum],
) -> frozenset[str]:
    """returns the values of an Enum class as a frozenset

    Parameters
    ----------
    enum_cls : Any
        Enum class

    Returns
    -------
    frozenset
        values of the Enum class
    """
    return frozenset(val.value for val in enum)
