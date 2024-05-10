from __future__ import annotations
from typing import TypeAlias, Any
from collections.abc import Iterator


FlattableObject: TypeAlias = (
    list['FlattableObject' | Any] | tuple['FlattableObject' | Any, ...] | set['FlattableObject' | Any]
)

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
