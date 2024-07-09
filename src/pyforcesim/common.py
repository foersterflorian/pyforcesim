from __future__ import annotations

import shutil
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, Type

from pyforcesim import datetime as pyf_dt
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
    enum_class: Type[StrEnum],
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
    return frozenset(val.value for val in enum_class)


def create_folder(
    path: Path,
    delete_existing: bool = False,
) -> None:
    if delete_existing and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_timestamp(
    with_time: bool = False,
) -> str:
    dt = pyf_dt.current_time_tz(pyf_dt.TIMEZONE_CEST)
    if with_time:
        return dt.strftime(r'%Y-%m-%d--%H-%M-%S')
    return dt.strftime(r'%Y-%m-%d')


def prepare_save_paths(
    base_folder: str | None,
    target_folder: str | None,
    filename: str | None,
    suffix: str | None,
    include_timestamp: bool = False,
    create_folder: bool = False,
) -> Path:
    if not any((base_folder, target_folder, filename, suffix)):
        raise ValueError('All parameters >>None<<')
    if not (
        all(x is None for x in (filename, suffix))
        or all(x is not None for x in (filename, suffix))
    ):
        raise ValueError('Filename and suffix must be provided together')
    if include_timestamp and filename is None:
        raise ValueError('Timestamp only with filename')
    if base_folder is None:
        base_folder = ''
    if target_folder is None:
        target_folder = ''
    if filename is None:
        filename = ''
    if include_timestamp:
        timestamp = get_timestamp(with_time=True)
        filename = f'{timestamp}_{filename}'
    if suffix is None:
        suffix = ''
    elif suffix is not None and suffix == '.':
        raise ValueError('Suffix can not be just dot.')
    elif suffix is not None and not suffix.startswith('.'):
        suffix = f'.{suffix}'

    pth_parent = (Path.cwd() / base_folder / target_folder).resolve()
    if create_folder and not pth_parent.exists():
        pth_parent.mkdir(parents=True)

    save_pth = (pth_parent / filename).with_suffix(suffix)

    return save_pth
