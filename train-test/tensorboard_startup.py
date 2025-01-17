import shlex
import subprocess
import time
import webbrowser
from threading import Thread
from typing import Final

import pyforcesim.loggers
from pyforcesim import common

pyforcesim.loggers.disable_logging()


USE_TRAIN_CONFIG: Final[bool] = True
if USE_TRAIN_CONFIG:
    from train import BASE_FOLDER, FOLDER_TB  # type: ignore
else:
    # EXPERIMENT_FOLDER: Final[str] = '2024-07-23-10__1-5-20__ConstIdeal__Util'
    # EXPERIMENT_FOLDER: Final[str] = '2024-07-23-10__1-5-30__ConstIdeal__Util'
    # EXPERIMENT_FOLDER: Final[str] = '2024-07-23-10__1-5-50__ConstIdeal__Util'
    EXPERIMENT_FOLDER: Final[str] = '2024-07-23-11__1-5-70__ConstIdeal__Util'
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
    run_cmd = shlex.split(command)
    subprocess.run(run_cmd)


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
