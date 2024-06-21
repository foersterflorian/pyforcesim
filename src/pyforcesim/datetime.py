"""Advanced time handling for simulation runs"""

from __future__ import annotations

import datetime
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from datetime import tzinfo as TZInfo

from pandas import DataFrame

from pyforcesim import common
from pyforcesim.constants import (
    TIMEZONE_CEST,
    TIMEZONE_UTC,
    TimeUnitsTimedelta,
)


def timedelta_from_val(
    val: float,
    time_unit: TimeUnitsTimedelta,
) -> Timedelta:
    """create Python timedelta object by choosing time value and time unit

    Parameters
    ----------
    val : float
        duration
    time_unit : str
        target time unit

    Returns
    -------
    Timedelta
        timedelta object corresponding to the given values

    Raises
    ------
    ValueError
        if chosen time unit not implemented
    """
    try:
        TimeUnitsTimedelta(time_unit)
    except ValueError:
        allowed_time_units = common.enum_str_values_as_frzset(TimeUnitsTimedelta)
        raise ValueError(
            f'Time unit >>{time_unit}<< not supported. Choose from {allowed_time_units}'
        )
    else:
        kwargs = {time_unit: val}
        return datetime.timedelta(**kwargs)


def dt_with_tz_UTC(
    *args,
    **kwargs,
) -> Datetime:
    return Datetime(*args, **kwargs, tzinfo=TIMEZONE_UTC)


def round_td_by_seconds(
    td: Timedelta,
    round_to_next_seconds: int = 1,
) -> Timedelta:
    """round timedelta object to the next full defined seconds

    Parameters
    ----------
    td : Timedelta
        timedelta object to be rounded
    round_to_next_seconds : int, optional
        number of seconds to round to, by default 1

    Returns
    -------
    Timedelta
        rounded timedelta object
    """
    total_seconds = td.total_seconds()
    rounded_seconds = round(total_seconds / round_to_next_seconds) * round_to_next_seconds
    return Timedelta(seconds=rounded_seconds)


def current_time_tz(
    tz: TZInfo = TIMEZONE_UTC,
    cut_microseconds: bool = False,
) -> Datetime:
    """current time as datetime object with
    associated time zone information (UTC by default)

    Parameters
    ----------
    tz : TZInfo, optional
        time zone information, by default TIMEZONE_UTC

    Returns
    -------
    Datetime
        datetime object with corresponding time zone
    """
    if cut_microseconds:
        return Datetime.now(tz=tz).replace(microsecond=0)
    else:
        return Datetime.now(tz=tz)


def add_timedelta_with_tz(
    starting_dt: Datetime,
    td: Timedelta,
) -> Datetime:
    """time-zone-aware calculation of an end point in time
    with a given timedelta

    Parameters
    ----------
    starting_dt : Datetime
        starting point in time
    td : Timedelta
        duration as timedelta object

    Returns
    -------
    Datetime
        time-zone-aware end point
    """

    if starting_dt.tzinfo is None:
        # no time zone information
        raise ValueError('The provided starting date does not contain time zone information.')
    else:
        # obtain time zone information from starting datetime object
        tz_info = starting_dt.tzinfo

    # transform starting point in time to utc
    dt_utc = starting_dt.astimezone(TIMEZONE_UTC)
    # all calculations are done in UTC
    # add duration
    ending_dt_utc = dt_utc + td
    # transform back to previous time zone
    ending_dt = ending_dt_utc.astimezone(tz=tz_info)

    return ending_dt


def validate_dt_UTC(
    dt: Datetime,
) -> None:
    """_summary_

    Parameters
    ----------
    dt : Datetime
        datetime object to be checked for available UTC time zone
        information

    Raises
    ------
    ValueError
        if no UTC time zone information is found
    """

    if dt.tzinfo != TIMEZONE_UTC:
        raise ValueError(
            f'Datetime object {dt} does not contain ' 'necessary UTC time zone information'
        )


def dt_to_timezone(
    dt: Datetime,
    target_tz: TZInfo = TIMEZONE_CEST,
) -> Datetime:
    """_summary_

    Parameters
    ----------
    dt : Datetime
        datetime with time zone information
    target_tz : TZInfo, optional
        target time zone information, by default TIMEZONE_CEST

    Returns
    -------
    Datetime
        datetime object adjusted to given local time zone

    Raises
    ------
    RuntimeError
        if datetime object does not contain time zone information
    """

    if dt.tzinfo is None:
        # no time zone information
        raise ValueError('The provided starting date does not contain time zone information.')
    # transform to given target time zone
    dt_local_tz = dt.astimezone(tz=target_tz)

    return dt_local_tz


def cut_dt_microseconds(
    dt: Datetime,
) -> Datetime:
    return dt.replace(microsecond=0)


# TODO rework after change to database usage
def get_date_cols_from_db(
    db: DataFrame,
) -> list[str]:
    target_cols: list[str] = []

    for col in db.columns:
        if 'date' in col and 'deviation' not in col:
            target_cols.append(col)

    return target_cols.copy()


def adjust_db_dates_local_tz(
    db: DataFrame,
    tz: TZInfo = TIMEZONE_CEST,
) -> DataFrame:
    db = db.copy()
    # obtain date columns from database
    date_cols = get_date_cols_from_db(db=db)
    # adjust UTC times to local time zone provided
    dates_only = db[date_cols]
    dates_only = dates_only.map(dt_to_timezone, na_action='ignore', target_tz=tz)
    db[date_cols] = dates_only

    return db.copy()
