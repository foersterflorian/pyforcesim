import pandas as pd
import sqlalchemy as sql
from pandas import DataFrame
from sqlalchemy import Column, ForeignKey, Table

from pyforcesim.constants import DB_ECHO, DB_HANDLE, DEFAULT_DATETIME
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

logical_queues = Table(
    'logical_queues',
    metadata_obj,
    Column('sys_id', sql.Integer, primary_key=True),
    Column('custom_id', sql.String, nullable=False, unique=True),
    Column('name', sql.String),
)

resources = Table(
    'resources',
    metadata_obj,
    Column('sys_id', sql.Integer, primary_key=True),
    Column('stat_group_id', ForeignKey('station_groups.sys_id'), nullable=False),
    Column('logical_queue_id', ForeignKey('logical_queues.sys_id'), nullable=False),
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
    Column('total_order_time', sql.Interval, nullable=False),
    Column('total_proc_time', sql.Interval, nullable=False),
    Column('total_setup_time', sql.Interval, nullable=False),
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
    Column('order_time', sql.Interval, nullable=False),
    Column('proc_time', sql.Interval, nullable=False),
    Column('setup_time', sql.Interval, nullable=False),
    Column('creation_date', sql.DateTime, nullable=False),
    Column('release_date', sql.DateTime, default=None),
    Column('planned_starting_date', sql.DateTime, default=None),
    Column('actual_starting_date', sql.DateTime, default=None),
    Column('starting_date_deviation', sql.Interval, default=None),
    Column('planned_ending_date', sql.DateTime, default=None),
    Column('actual_ending_date', sql.DateTime, default=None),
    Column('ending_date_deviation', sql.Interval, default=None),
    Column('lead_time', sql.Interval, default=None),
    Column('slack_init', sql.Float, default=None),
    Column('slack_end', sql.Float, default=None),
    Column('slack_lower_bound', sql.Float, default=None),
    Column('slack_upper_bound', sql.Float, default=None),
)


def get_engine(
    db_handle: str | None,
) -> sql.Engine:
    if db_handle is None:
        db_handle = DB_HANDLE
    engine = sql.create_engine(db_handle, echo=DB_ECHO)
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


def parse_database_to_dataframe(
    database: sql.Table,
    db_engine: sql.Engine,
    datetime_parse_info: PandasDateColParseInfo | None = None,
    timedelta_cols: PandasTimedeltaCols | None = None,
) -> DataFrame:
    db_df = pd.read_sql_table(
        database.name,
        db_engine,
        parse_dates=datetime_parse_info,
    )
    if timedelta_cols is not None:
        db_df[timedelta_cols] = db_df.loc[:, timedelta_cols] - DEFAULT_DATETIME

    return db_df


def parse_sql_query_to_dataframe(
    query_select: sql.Select,
    engine: sql.Engine,
    index_col: str | None = None,
    datetime_parse_info: PandasDateColParseInfo | None = None,
    timedelta_cols: PandasTimedeltaCols | None = None,
) -> DataFrame:
    db_df = pd.read_sql_query(
        sql=query_select,
        con=engine,
        index_col=index_col,
        parse_dates=datetime_parse_info,
    )
    if timedelta_cols is not None:
        try:
            db_df[timedelta_cols] = db_df.loc[:, timedelta_cols] - DEFAULT_DATETIME
        except TypeError:
            # catch inconsistent Pandas behaviour:
            # reading from tables parses timedelta values as datetime relative
            # to the default datetime, but when parsing by query is used these entries
            # are automatically parsed as timedelta, so no further adaption is necessary
            pass

    return db_df
