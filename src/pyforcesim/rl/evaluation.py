from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyforcesim import datetime as pyf_dt
from pyforcesim.constants import TimeUnitsTimedelta

if TYPE_CHECKING:
    from pyforcesim.types import PlotlyFigure, Timedelta


def load_exported_db_from_pickle(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Provided path does not exist: {path}')

    return pd.read_pickle(path)


def _validate_dbs_on_properties(
    db_1: pd.DataFrame,
    db_2: pd.DataFrame,
    field: str,
) -> None:
    db_1_fields = set(db_1.columns)
    db_2_fields = set(db_2.columns)
    if db_1_fields - db_2_fields:
        raise KeyError('The provided databases do not contain the same fields.')
    elif field not in db_1_fields:
        raise KeyError(
            f'Provided property >>{field}<< does not exist in the provided databases.'
        )


def merge_dbs_on_property(
    db_1: pd.DataFrame,
    db_2: pd.DataFrame,
    field: str,
    title_1: str = 'Agent',
    title_2: str = 'Benchmark',
) -> pd.DataFrame:
    _validate_dbs_on_properties(db_1, db_2, field)
    # check for timedelta:
    NORM_TD: Final[Timedelta] = pyf_dt.timedelta_from_val(1, TimeUnitsTimedelta.HOURS)
    is_dt_like = hasattr(db_1[field], 'dt')
    is_timedelta: bool = False
    if is_dt_like:
        try:
            db_1[field].dt.days
            is_timedelta = True
        except AttributeError:
            pass

    data_1 = db_1[field].copy()
    data_1_labels = [title_1] * len(data_1)
    data_2 = db_2[field].copy()
    data_2_labels = [title_2] * len(data_2)

    if is_timedelta:
        data_1 = data_1 / NORM_TD  # type: ignore
        data_2 = data_2 / NORM_TD  # type: ignore

    df_data_1 = pd.DataFrame({field: data_1, 'label': data_1_labels})
    df_data_2 = pd.DataFrame({field: data_2, 'label': data_2_labels})

    # concatenation
    return pd.concat([df_data_1, df_data_2], ignore_index=True)


def boxplot(
    db_1: pd.DataFrame,
    db_2: pd.DataFrame,
    field: str,
    title_1: str = 'Agent',
    title_2: str = 'Benchmark',
    height: int = 600,
) -> PlotlyFigure:
    df_combined = merge_dbs_on_property(
        db_1,
        db_2,
        field=field,
        title_1=title_1,
        title_2=title_2,
    )
    fig = px.box(df_combined, y=field, x='label')
    title = f'Boxplot - property: {field}'
    fig.update_layout(title=title, height=height)

    return fig


def histogram(
    db_1: pd.DataFrame,
    db_2: pd.DataFrame,
    field: str,
    title_1: str = 'Agent',
    title_2: str = 'Benchmark',
    height: int = 800,
) -> PlotlyFigure:
    _validate_dbs_on_properties(db_1, db_2, field)
    data_1 = db_1[field].copy()
    data_2 = db_2[field].copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(title_1, title_2, title_2, title_1),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Histogram(x=data_1, name=title_1, marker=dict(color='#0099ff')), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=data_2, name=title_2, marker=dict(color='#00cc66')),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=data_2, name=title_2, marker=dict(color='#00cc66')),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=data_1, name=title_1, marker=dict(color='#0099ff')), row=2, col=2
    )

    fig.add_vline(0, line_width=2, line_dash='dash')
    title = f'Histogram - property: {field}'
    fig.update_layout(title=title, height=height)

    return fig
