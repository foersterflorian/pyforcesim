from dash_extensions.enrich import Dash, DashProxy, html, dcc, Output, Input
from dash_extensions import WebSocket
from plotly.graph_objs._figure import Figure as PlotlyFigure
import plotly.io
import webbrowser
import time
import threading
import logging
from .websocket_server import WS_HOST, WS_PORT, WS_ROUTE

# ** configuration
HOST: str = '127.0.0.1'
PORT: int = 8081
URL: str = f'http://{HOST}:{PORT}'
WS_URL: str = f'ws://{WS_HOST}:{WS_PORT}/{WS_ROUTE}'


# ** Dash Application
app = DashProxy(__name__, prevent_initial_callbacks=True)

gantt_chart = dcc.Graph(id='gantt_chart')
gantt_chart.figure = PlotlyFigure()

app.layout = html.Div([
    html.H1(children='Dashboard SimRL', style={'textAlign':'center'}),
    gantt_chart,
    #WebSocket(id="ws", url="ws://127.0.0.1:5000/gantt_chart"),
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
    #print(f"{type(gantt_chart_json)=}")
    gantt_chart = plotly.io.from_json(gantt_chart_json)
    #print(f"Response from websocket")
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