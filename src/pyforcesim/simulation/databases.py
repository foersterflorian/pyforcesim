import sqlalchemy as sql
from sqlalchemy import Column, ForeignKey, Table

from pyforcesim.constants import DB_ECHO, DB_HANDLE
from pyforcesim.types import (
    PandasDateColParseInfo,
    PandasDatetimeCols,
    PandasTimedeltaCols,
)

metadata_obj = sql.MetaData()


production_areas = Table(
    'prod_areas',
    metadata_obj,
    Column('sys_id', sql.Integer, primary_key=True),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('name', sql.String),
    Column('contains_proc_stations', sql.Boolean, default=False),
)

station_groups = Table(
    'station_groups',
    metadata_obj,
    Column('sys_id', sql.Integer, primary_key=True),
    Column('prod_area_id', ForeignKey('prod_areas.sys_id'), nullable=False),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('name', sql.String),
    Column('contains_proc_stations', sql.Boolean, default=False),
)

resources = Table(
    'resources',
    metadata_obj,
    Column('sys_id', sql.Integer, primary_key=True),
    Column('stat_group_id', ForeignKey('station_groups.sys_id'), nullable=False),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('name', sql.String),
    Column('type', sql.String, nullable=False),
    Column('state', sql.String, nullable=False),
)


jobs = Table(
    'jobs',
    metadata_obj,
    Column('load_id', sql.Integer, primary_key=True),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('type', sql.String, nullable=False),
    Column('prio', sql.Integer, default=None),
    Column('state', sql.String, nullable=False),
    Column('total_proc_time', sql.Interval, nullable=False),
    Column('creation_date', sql.DateTime, nullable=False),
    Column('release_date', sql.DateTime, default=None),
    Column('planned_starting_date', sql.DateTime, default=None),
    Column('actual_starting_date', sql.DateTime, default=None),
    Column('starting_date_deviation', sql.Interval, default=None),
    Column('planned_ending_date', sql.DateTime, default=None),
    Column('actual_ending_date', sql.DateTime, default=None),
    Column('ending_date_deviation', sql.Interval, default=None),
    Column('lead_time', sql.Interval, default=None),
)


operations = Table(
    'operations',
    metadata_obj,
    Column('load_id', sql.Integer, primary_key=True),
    Column('job_id', ForeignKey('jobs.load_id'), nullable=False),
    Column('execution_sys_id', ForeignKey('prod_areas.sys_id'), nullable=False),
    Column('station_group_sys_id', ForeignKey('station_groups.sys_id'), nullable=True),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('target_station_sys_id', ForeignKey('resources.sys_id'), default=None),
    Column('prio', sql.Integer, default=None),
    Column('state', sql.String, nullable=False),
    Column('proc_time', sql.Interval, nullable=False),
    Column('setup_time', sql.Interval, nullable=False),
    Column('order_time', sql.Interval, nullable=False),
    Column('creation_date', sql.DateTime, nullable=False),
    Column('release_date', sql.DateTime, default=None),
    Column('planned_starting_date', sql.DateTime, default=None),
    Column('actual_starting_date', sql.DateTime, default=None),
    Column('starting_date_deviation', sql.Interval, default=None),
    Column('planned_ending_date', sql.DateTime, default=None),
    Column('actual_ending_date', sql.DateTime, default=None),
    Column('ending_date_deviation', sql.Interval, default=None),
    Column('lead_time', sql.Interval, default=None),
)


def get_engine() -> sql.Engine:
    engine = sql.create_engine(DB_HANDLE, echo=DB_ECHO)
    return engine


def pandas_date_col_parser(
    database: sql.Table,
) -> tuple[PandasDateColParseInfo, PandasDatetimeCols, PandasTimedeltaCols]:
    date_cols_pandas: PandasDateColParseInfo = {}
    datetime_cols: PandasDatetimeCols = []
    timedelta_cols: PandasTimedeltaCols = []
    for col in database.c:
        if isinstance(col.type, sql.DateTime):
            date_cols_pandas[col.name] = {'utc': True}
            datetime_cols.append(col.name)
        elif isinstance(col.type, sql.Interval):
            date_cols_pandas[col.name] = {'utc': True}
            timedelta_cols.append(col.name)

    return date_cols_pandas, datetime_cols, timedelta_cols
