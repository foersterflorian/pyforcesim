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
    db_agent: pd.DataFrame,
    db_bench: pd.DataFrame,
    field: str,
) -> None:
    agent_fields = set(db_agent.columns)
    bench_fields = set(db_bench.columns)
    if agent_fields - bench_fields:
        raise KeyError('The provided databases do not contain the same fields.')
    elif field not in agent_fields:
        raise KeyError(
            f'Provided property >>{field}<< does not exist in the provided databases.'
        )


def merge_dbs_on_property(
    db_agent: pd.DataFrame,
    db_bench: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    _validate_dbs_on_properties(db_agent, db_bench, field)
    # check for timedelta:
    NORM_TD: Final[Timedelta] = pyf_dt.timedelta_from_val(1, TimeUnitsTimedelta.HOURS)
    is_dt_like = hasattr(db_agent[field], 'dt')
    is_timedelta: bool
    if is_dt_like:
        try:
            db_agent[field].dt.days
            is_timedelta = True
        except AttributeError:
            is_timedelta = False

    # benchmark
    data_bench = db_bench[field].copy()
    bench_labels = ['bench'] * len(data_bench)
    # agent
    data_agent = db_agent[field].copy()
    agent_labels = ['agent'] * len(data_agent)

    if is_timedelta:
        data_bench = data_bench / NORM_TD  # type: ignore
        data_agent = data_agent / NORM_TD  # type: ignore

    df_slack_bench = pd.DataFrame({field: data_bench, 'label': bench_labels})
    df_slack_agent = pd.DataFrame({field: data_agent, 'label': agent_labels})
    # concatenation
    return pd.concat([df_slack_agent, df_slack_bench], ignore_index=True)


def boxplot(
    db_agent: pd.DataFrame,
    db_bench: pd.DataFrame,
    field: str,
    height: int = 600,
) -> PlotlyFigure:
    df_combined = merge_dbs_on_property(db_agent, db_bench, field=field)
    fig = px.box(df_combined, y=field, x='label')
    title = f'Boxplot - property: {field}'
    fig.update_layout(title=title, height=height)

    return fig


def histogram(
    db_agent: pd.DataFrame,
    db_bench: pd.DataFrame,
    field: str,
    height: int = 800,
) -> PlotlyFigure:
    _validate_dbs_on_properties(db_agent, db_bench, field)
    data_agent = db_agent[field].copy()
    data_bench = db_bench[field].copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Agent', 'Benchmark', 'Benchmark', 'Agent'),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Histogram(x=data_agent, name='Agent', marker=dict(color='#0099ff')), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=data_bench, name='Benchmark', marker=dict(color='#00cc66')),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=data_bench, name='Benchmark', marker=dict(color='#00cc66')),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=data_agent, name='Agent', marker=dict(color='#0099ff')), row=2, col=2
    )

    fig.add_vline(0, line_width=2, line_dash='dash')
    title = f'Histogram - property: {field}'
    fig.update_layout(title=title, height=height)

    return fig
