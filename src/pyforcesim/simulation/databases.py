import sqlalchemy as sql
from sqlalchemy import Column, ForeignKey, Table

from pyforcesim.constants import DB_HANDLE, DB_ECHO

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


def get_engine() -> sql.Engine:
    engine = sql.create_engine(DB_HANDLE, echo=DB_ECHO)
    return engine
