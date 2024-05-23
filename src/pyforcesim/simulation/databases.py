# module to provide databases for simulation runs
# not specified yet

import os
import sqlite3 as sql
from datetime import date, datetime
from pathlib import Path
from typing import Final

DB_ROOT: Final[str] = 'databases'


def adapt_date_iso(val: date) -> str:
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_iso(val: datetime) -> str:
    """Adapt datetime.datetime to ISO 8601 date."""
    return val.isoformat()


def convert_date(val: bytes) -> date:
    """Convert ISO 8601 date to datetime.date object."""
    return date.fromisoformat(val.decode())


def convert_datetime(val: bytes) -> datetime:
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.fromisoformat(val.decode())


sql.register_adapter(date, adapt_date_iso)
sql.register_adapter(datetime, adapt_datetime_iso)
sql.register_converter('date', convert_date)
sql.register_converter('datetime', convert_datetime)


# connection with: detect_types=sqlite3.PARSE_DECLTYPES
# always use the declared type as reference for type conversion


class Database:
    def __init__(
        self,
        name: str,
    ) -> None:
        # create database folder
        cwd = Path.cwd()
        db_folder = cwd / DB_ROOT
        if not db_folder.exists():
            os.makedirs(db_folder)

        self._name = name
        self._path = (db_folder / name).with_suffix('.db')

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path
