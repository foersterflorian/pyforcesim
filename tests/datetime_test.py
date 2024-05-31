from datetime import UTC, datetime, timedelta

import pytest
from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.datetime import TIMEZONE_CEST


def test_make_dt_with_UTC(dt_manager):
    year = 2024
    month = 3
    day = 28
    hour = 3
    minute = 0
    dt_mgr = dt_manager.dt_with_tz_UTC(year, month, day, hour, minute)
    dt = datetime(year, month, day, hour, minute, tzinfo=UTC)
    assert dt == dt_mgr


@pytest.mark.parametrize(
    'time_unit, expected',
    [
        ('hours', timedelta(hours=2.0)),
        ('minutes', timedelta(minutes=2.0)),
        ('seconds', timedelta(seconds=2.0)),
        ('milliseconds', timedelta(milliseconds=2.0)),
        ('microseconds', timedelta(microseconds=2.0)),
        (TimeUnitsTimedelta.HOURS, timedelta(hours=2.0)),
        (TimeUnitsTimedelta.MINUTES, timedelta(minutes=2.0)),
    ],
)
def test_timedelta_from_val(dt_manager, time_unit, expected):
    val = 2.0
    td = dt_manager.timedelta_from_val(val, time_unit)
    assert td == expected


def test_timedelta_from_val_err(dt_manager):
    val = 2.0
    time_unit = 'years'
    with pytest.raises(ValueError):
        dt_manager.timedelta_from_val(val, time_unit)


def test_round_td_by_seconds(dt_manager):
    hours = 2.0
    minutes = 30.0
    seconds = 30.0
    microseconds = 600
    td = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    rounded_td = dt_manager.round_td_by_seconds(td, round_to_next_seconds=1)
    assert rounded_td == timedelta(hours=2.0, minutes=30.0, seconds=30.0)


def test_add_timedelta_err(dt_manager):
    year = 2024
    month = 3
    day = 30
    hour = 3
    minute = 0
    dt = datetime(year, month, day, hour, minute)
    td = timedelta(hours=2.0)
    with pytest.raises(ValueError):
        dt_manager.add_timedelta_with_tz(dt, td)


def test_add_timedelta_with_tz(dt_manager):
    year = 2024
    month = 3
    day = 30
    hour = 23
    minute = 0
    dt = datetime(year, month, day, hour, minute, tzinfo=TIMEZONE_CEST)
    td = timedelta(hours=6.0)
    new_dt = dt_manager.add_timedelta_with_tz(dt, td)
    assert new_dt == datetime(2024, 3, 31, 6, 0, tzinfo=TIMEZONE_CEST)


def test_validate_UTC(dt_manager):
    dt = datetime(2024, 3, 30, 0, 0, tzinfo=TIMEZONE_CEST)
    with pytest.raises(ValueError):
        dt_manager.validate_dt_UTC(dt)


def test_tz_conversion(dt_manager):
    dt = datetime(2024, 3, 30, 2, 0, tzinfo=UTC)
    new_dt = dt_manager.dt_to_timezone(dt, TIMEZONE_CEST)
    assert new_dt == datetime(2024, 3, 30, 3, tzinfo=TIMEZONE_CEST)


def test_validate_timezone_info_conversion(dt_manager):
    dt = datetime(2024, 3, 30, 2, 0)
    with pytest.raises(ValueError):
        dt_manager.dt_to_timezone(dt, TIMEZONE_CEST)


def test_cut_microseconds(dt_manager):
    dt = datetime(2024, 3, 30, 2, 0, 0, 600)
    new_dt = dt_manager.cut_dt_microseconds(dt)
    assert new_dt == datetime(2024, 3, 30, 2, 0, 0, 0)
