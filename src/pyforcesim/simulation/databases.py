# module to provide databases for simulation runs
# not specified yet

import re
import sqlite3 as sql
from collections.abc import Sequence
from datetime import date as Date
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from itertools import product
from pathlib import Path
from typing import Any, Final, cast

from pyforcesim.constants import DB_DATA_TYPES, DB_ROOT, DB_SUPPORTED_COL_CONSTRAINTS
from pyforcesim.errors import CommonSQLError
from pyforcesim.loggers import databases as logger
from pyforcesim.types import (
    DBColumnDeclaration,
    ForeignKeyInfo,
    SQLiteColumnDescription,
)

DB_INJECTION_PATTERN: Final[str] = (
    r'(^[_0-9]+)|[^\w ]|(true|false|select|where|drop|delete|create)'
)

db_col_combinations = product(DB_DATA_TYPES, DB_SUPPORTED_COL_CONSTRAINTS)
db_data_types_additional = {' '.join(entry) for entry in db_col_combinations}
DB_DATA_TYPES_ALLOWED: Final[frozenset[str]] = frozenset(
    DB_DATA_TYPES.union(db_data_types_additional)
)


def adapt_date_iso(val: Date) -> str:
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_iso(val: Datetime) -> str:
    """Adapt datetime.datetime to ISO 8601 date."""
    return val.isoformat()


def adapt_timedelta(td: Timedelta) -> str:
    return f'{td.days},{td.seconds},{td.microseconds}'


def adapt_bool(val: bool) -> int:
    return int(val)


def convert_date(val: bytes) -> Date:
    """Convert ISO 8601 date to datetime.date object."""
    return Date.fromisoformat(val.decode())


def convert_datetime(val: bytes) -> Datetime:
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return Datetime.fromisoformat(val.decode())


def convert_timedelta(val: bytes) -> Timedelta:
    days, secs, mic_secs = tuple(map(int, val.split(b',')))
    return Timedelta(days=days, seconds=secs, microseconds=mic_secs)


def convert_bool(val: bytes) -> bool:
    return bool(int(val))


sql.register_adapter(Date, adapt_date_iso)
sql.register_adapter(Datetime, adapt_datetime_iso)
sql.register_adapter(Timedelta, adapt_timedelta)
sql.register_adapter(bool, adapt_bool)
sql.register_converter('DATE', convert_date)
sql.register_converter('DATETIME', convert_datetime)
sql.register_converter('TIMEDELTA', convert_timedelta)
sql.register_converter('BOOLEAN', convert_bool)


# connection with: detect_types=sqlite3.PARSE_DECLTYPES
# always use the declared type as reference for type conversion


class Database:
    def __init__(
        self,
        name: str,
        delete_existing: bool = False,
        memory_only: bool = False,
    ) -> None:
        # create database folder
        cwd = Path.cwd()
        db_folder = cwd / DB_ROOT
        if not db_folder.exists():
            db_folder.mkdir()
        # properties
        self._name = name
        self.memory_only = memory_only
        self._path: Path | None = None
        if not self.memory_only:
            self._path = (db_folder / name).with_suffix('.db')
            # deletion
            if delete_existing and self._path.exists():
                self._path.unlink()
        # connections
        self.con: sql.Connection | None = None
        self.get_connection()
        # query control
        self._check_query_pattern = re.compile(DB_INJECTION_PATTERN, flags=re.IGNORECASE)

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def query_injection_pattern(self) -> re.Pattern:
        return self._check_query_pattern

    def _clean_query_injection(
        self,
        value: str,
    ) -> str:
        matches = self._check_query_pattern.search(value)
        if matches is not None:
            raise ValueError(f'Unallowed characters in value {value}.')

        return value

    def _format_column_declaration(
        self,
        columns: DBColumnDeclaration,
    ) -> str:
        col_def: list[str] = []
        for col_name, col_type in columns.items():
            col_type = col_type.upper()
            if col_type not in DB_DATA_TYPES_ALLOWED:
                raise ValueError(
                    (
                        f'Column type {col_type} not allowed. '
                        f'Must be one of: {DB_DATA_TYPES_ALLOWED}'
                    )
                )
            col_name = self._clean_query_injection(col_name)
            col_type = self._clean_query_injection(col_type)
            col_def.append(f'{col_name} {col_type}')

        col_query = ', '.join(col_def)

        return col_query

    def _add_foreign_key_constraint(
        self,
        col_query: str,
        key_col: str,
        ref_table: str,
        ref_col: str,
    ) -> str:
        key_col = self._clean_query_injection(key_col)
        ref_table = self._clean_query_injection(ref_table)
        ref_col = self._clean_query_injection(ref_col)
        fk_definition = f'FOREIGN KEY ({key_col}) REFERENCES {ref_table}({ref_col})'
        query = f'{col_query}, {fk_definition}'

        return query

    def create_foreign_key_info(
        self,
        column: str,
        ref_table: str,
        ref_column: str,
    ) -> ForeignKeyInfo:
        return {
            'column': column,
            'ref_table': ref_table,
            'ref_column': ref_column,
        }

    def get_connection(self) -> sql.Connection:
        if not self.memory_only:
            if self.path is None:
                raise ValueError('No path for database file provided.')
            self.con = sql.connect(self.path, detect_types=sql.PARSE_DECLTYPES)
        else:
            self.con = sql.connect(':memory:', detect_types=sql.PARSE_DECLTYPES)
        # explicitly enable foreign key constraints
        with self.con as con:
            con.execute('PRAGMA foreign_keys = ON')

        return self.con

    def close_connection(self) -> None:
        if self.con is None:
            logger.warning('No connection to close.')
        else:
            self.con.commit()
            self.con.close()
            self.con = None

    def get_columns(
        self,
        table_name: str,
    ) -> list[SQLiteColumnDescription]:
        table_name = self._clean_query_injection(table_name)
        query = f'PRAGMA table_info({table_name})'
        if self.con is None:
            raise ValueError('No connection to database established.')
        with self.con as con:
            res = con.execute(query)
            columns = cast(list[SQLiteColumnDescription], res.fetchall())r

        if not columns:
            raise CommonSQLError(
                f'Column retrieval failed. Maybe table >>{table_name}<< does not exist.'
            )

        return columns

    # !! dangerous: only use for testing
    # TODO remove later
    def execute_query(
        self,
        query: str,
    ) -> list[tuple[Any, ...]] | None:
        if self.con is None:
            raise ValueError('No connection to database established.')
        with self.con as con:
            res = con.execute(query)
            response = res.fetchall()

        if response:
            return response

    def create_table(
        self,
        table_name: str,
        columns: DBColumnDeclaration,
        *,
        foreign_key_info: ForeignKeyInfo | None = None,
    ) -> None:
        table_name = self._clean_query_injection(table_name)
        col_query = self._format_column_declaration(columns)

        if foreign_key_info is not None:
            col_query = self._add_foreign_key_constraint(
                col_query,
                foreign_key_info['column'],
                foreign_key_info['ref_table'],
                foreign_key_info['ref_column'],
            )

        logger.info(f'Creating table {table_name}...')
        query = f'CREATE TABLE IF NOT EXISTS {table_name} ({col_query})'
        logger.debug(f'Query: {query}.')
        try:
            if self.con is None:
                raise ValueError('No connection to database established.')
            with self.con as con:
                con.execute(query)
        except Exception as error:
            raise error
        else:
            logger.info(f'Table {table_name} created successfully.')

    def prepare_insertion_query(
        self,
        table_name: str,
    ) -> tuple[str, int]:
        # query injection check done in ``get_columns``
        columns = self.get_columns(table_name)
        num_cols = len(columns)
        placeholders = ', '.join('?' for _ in columns)
        query = f'INSERT INTO {table_name} VALUES ({placeholders})'
        logger.debug(f'Prepared insertion query: {query}.')
        return query, num_cols

    def prepare_insertion(
        self,
        table_name: str,
        data: tuple[Any, ...],
    ) -> str:
        table_name = self._clean_query_injection(table_name)
        query, num_cols = self.prepare_insertion_query(table_name)
        if len(data) != num_cols:
            raise ValueError(
                (
                    f'Number of data elements >>{len(data)}<< does not '
                    f'match number of columns >>{num_cols}<<.'
                )
            )

        return query

    def insert(
        self,
        table_name: str,
        data: tuple[Any, ...],
    ) -> None:
        query = self.prepare_insertion(table_name, data)
        logger.debug(f'Inserting data into table {table_name} with {query=}.')
        try:
            if self.con is None:
                raise ValueError('No connection to database established.')
            with self.con as con:
                con.execute(query, data)
        except Exception as error:
            raise error
        else:
            logger.debug('Data inserted successfully.')

    def insert_many(
        self,
        table_name: str,
        data: Sequence[tuple[Any, ...]],
    ) -> None:
        query = self.prepare_insertion(table_name, data[0])
        logger.debug(f'Inserting data into table {table_name} with {query=}.')
        try:
            if self.con is None:
                raise ValueError('No connection to database established.')
            with self.con as con:
                con.executemany(query, data)
        except Exception as error:
            raise error
        else:
            logger.debug('Data inserted successfully.')


"""
database definitions, later moved to other place
# Infrastructure Manager
## Production Area
name = 'production_areas'
cols_props = {
    'id': 'INTEGER PRIMARY KEY',
    'custom_id': 'TEXT',
    'name': 'TEXT',
    'containing_proc_station': 'BOOLEAN',
}
## Station Groups
name = 'station_groups'
cols_props = {
    'id': 'INTEGER PRIMARY KEY',
    'production_area_id': 'INTEGER NOT NULL',
    'custom_id': 'TEXT',
    'name': 'TEXT',
    'containing_proc_station': 'BOOLEAN',
}
fk_info = db.create_foreign_key_info(
    column='production_area_id',
    ref_table='production_areas',
    ref_column='id',
)
## Resources
name = 'resources'
cols_props = {
    'id': 'INTEGER PRIMARY KEY',
    'station_group_id': 'INTEGER NOT NULL',
    'custom_id': 'TEXT',
    'name': 'TEXT',
    'type': 'TEXT',
    'state': 'TEXT',
}
fk_info = db.create_foreign_key_info(
    column='station_group_id',
    ref_table='station_groups',
    ref_column='id',
)
"""
