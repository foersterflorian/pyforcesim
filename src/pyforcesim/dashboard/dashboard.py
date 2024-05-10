from typing import Final
import webbrowser
import time
import threading

from dash_extensions.enrich import DashProxy, html, dcc, Output, Input
from dash_extensions import WebSocket
import plotly.io

from pyforcesim.dashboard.websocket_server import WS_HOST, WS_PORT, WS_ROUTE

# ** configuration
HOST: Final[str] = '127.0.0.1'
PORT: Final[str] = '8081'
URL: Final[str] = f'http://{HOST}:{PORT}'
WS_URL: Final[str] = f'ws://{WS_HOST}:{WS_PORT}/{WS_ROUTE}'


# ** Dash Application
app = DashProxy(__name__, prevent_initial_callbacks=True) # type: ignore (error in DashProxy definition)

# TODO remove
#gantt_chart = dcc.Graph(id='gantt_chart')
#gantt_chart.figure = PlotlyFigure()

app.layout = html.Div([
    html.H1(children='Dashboard SimRL', style={'textAlign':'center'}),
    dcc.Graph(id='gantt_chart'),
    WebSocket(id="ws", url=WS_URL),
])

# updating Gantt chart
@app.callback(
    Output("gantt_chart", "figure"),
    Input("ws", "message"),
)
def update_gantt_chart(
    message: dict[str, str],
):
    gantt_chart_json = message['data']
    gantt_chart = plotly.io.from_json(gantt_chart_json)
    return gantt_chart

# ** dashboard management
def start_webbrowser(
    url: str,
) -> None:
    time.sleep(1)
    webbrowser.open_new(url=url)

def start_dashboard() -> None:
    # open webbrowser to display dashboard
    webbrowser_thread = threading.Thread(target=start_webbrowser, args=(URL,))
    webbrowser_thread.start()
    # run dashboard app
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)
    # closing
    webbrowser_thread.join()


if __name__ == '__main__':
    start_dashboard()