import argparse
import shlex
import subprocess
import time
import webbrowser
from threading import Thread
from typing import Final

import pyforcesim.loggers
from pyforcesim import common
from pyforcesim.config import TB_EXP_FOLDER, TB_USE_TRAIN_CONFIG

pyforcesim.loggers.disable_logging()


if TB_USE_TRAIN_CONFIG:
    from train import BASE_FOLDER, FOLDER_TB  # type: ignore
else:
    BASE_FOLDER: Final[str] = f'results/{TB_EXP_FOLDER}'
    FOLDER_TB: Final[str] = 'tensorboard'

LOG_DIR = common.prepare_save_paths(BASE_FOLDER, FOLDER_TB, None, None)
assert LOG_DIR.exists(), 'Tensorboard path does not exist'

PORT: Final[int] = 6006


def start_tensorboard(
    port: int,
    expose_network: bool = False,
) -> None:
    command_parts: list[str] = [
        'pdm run',
        'tensorboard',
        f'--logdir="{LOG_DIR}"',
        f'--port={port}',
    ]
    if expose_network:
        command_parts.append('--bind_all')

    command = ' '.join(command_parts)
    print(f'Starting with command: >> {command}')
    run_cmd = shlex.split(command)
    subprocess.run(run_cmd)


def open_browser() -> None:
    time.sleep(6)
    url = f'http://localhost:{PORT}/'
    webbrowser.open_new(url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w',
        '--webbrowser',
        help='open TensorBoard page automatically',
        action='store_true',
    )
    parser.add_argument(
        '-p',
        '--port',
        help='expose TensorBoard port to network',
        type=int,
        default=PORT,
    )
    parser.add_argument(
        '-e',
        '--expose',
        help='expose TensorBoard port to network',
        action='store_true',
    )
    args = parser.parse_args()

    if args.webbrowser:
        webbrowser_thread = Thread(target=open_browser, daemon=True)
        webbrowser_thread.start()

    expose_network: bool = False
    if args.expose:
        expose_network = True

    try:
        start_tensorboard(args.port, expose_network)
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Tensorboard host stopped.')


if __name__ == '__main__':
    main()
