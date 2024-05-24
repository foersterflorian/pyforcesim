import asyncio
from asyncio import Queue as AsyncQueue
from collections.abc import AsyncIterator
from typing import Final

from quart import Quart, websocket

# ** network properties
WS_HOST: Final[str] = '127.0.0.1'
WS_PORT: Final[int] = 5000
WS_ROUTE: Final[str] = 'gantt_chart'


# publish-subscribe pattern based on Quart tutorial
# https://quart.palletsprojects.com/en/latest/tutorials/chat_tutorial.html
class Broker:
    def __init__(self) -> None:
        self.connections: set[AsyncQueue[str]] = set()

    async def publish(self, message: str) -> None:
        for connection in self.connections:
            await connection.put(message)

    async def subscribe(self) -> AsyncIterator[str]:
        connection: AsyncQueue[str] = asyncio.Queue()
        self.connections.add(connection)
        try:
            while True:
                yield await connection.get()
        finally:
            self.connections.remove(connection)


# ** app definition
app = Quart(__name__)
broker = Broker()


async def receive() -> None:
    while True:
        message = await websocket.receive()
        await broker.publish(message)


@app.websocket(f'/{WS_ROUTE}')
async def ws() -> None:
    task = asyncio.create_task(receive())
    try:
        async for message in broker.subscribe():
            await websocket.send(message)
    finally:
        task.cancel()
        await task


def start_websocket_server() -> None:
    app.run(host=WS_HOST, port=WS_PORT, debug=True)


# should not be called as main
if __name__ == '__main__':
    start_websocket_server()
