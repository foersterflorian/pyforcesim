import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from threading import Thread
from typing import Final

from pyforcesim import common

EXPERIMENT_FOLDER: Final[str] = '2024-07-22-01__1-5-15__ConstIdeal__Util'
BASE_FOLDER: Final[str] = f'results/{EXPERIMENT_FOLDER}'
FOLDER_TB: Final[str] = 'tensorboard'
LOG_DIR = common.prepare_save_paths(BASE_FOLDER, FOLDER_TB, None, None)
assert LOG_DIR.exists(), 'Tensorboard path does not exist'

PORT: Final[int] = 6006


def start_tensorboard() -> None:
    command_parts: list[str] = [
        'pdm run',
        'tensorboard',
        f'--logdir="{LOG_DIR}"',
        f'--port={PORT}',
    ]
    command = ' '.join(command_parts)
    print(f'Starting with command: >> {command}')
    subprocess.run(command)


def open_browser() -> None:
    time.sleep(6)
    url = f'http://localhost:{PORT}/'
    webbrowser.open_new(url)


def main() -> None:
    webbrowser_thread = Thread(target=open_browser, daemon=True)
    webbrowser_thread.start()
    start_tensorboard()


if __name__ == '__main__':
    main()
