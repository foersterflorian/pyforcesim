from pathlib import Path
from typing import Any, Final

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pyforcesim import common
from pyforcesim import datetime as pyf_dt
from pyforcesim.rl.gym_env import JSSEnv

OVERWRITE_FOLDERS: Final[bool] = False
CONTINUE_LEARNING: Final[bool] = True
CALC_ITERATIONS: Final[int] = 233472 // 2048

DATE = common.get_timestamp(with_time=False)
EXP_NUM: Final[str] = '01'
ENV_STRUCTURE: Final[str] = '1-3-7'
JOB_GEN_METHOD: Final[str] = 'ConstIdeal'
EXP_TYPE: Final[str] = f'{ENV_STRUCTURE}_{JOB_GEN_METHOD}'
FEEDBACK_MACHANISM: Final[str] = 'Util'
EXPERIMENT_FOLDER: Final[str] = (
    f'{DATE}-{EXP_NUM}__{ENV_STRUCTURE}__{JOB_GEN_METHOD}__{FEEDBACK_MACHANISM}'
)
BASE_FOLDER: Final[str] = f'results/{EXPERIMENT_FOLDER}'

FOLDER_TB: Final[str] = 'tensorboard'
FOLDER_MODEL_SAVEPOINTS: Final[str] = 'models'

MODEL: Final[str] = 'PPO_mask'
MODEL_BASE_NAME: Final[str] = f'pyf_sim_{MODEL}'
TIMESTEPS_PER_ITER: Final[int] = int(2048 * 1)
ITERATIONS: Final[int] = 100
ITERATIONS_TILL_SAVE: Final[int] = 1

FILENAME_TARGET_MODEL: Final[str] = '2024-06-24--19-35-38_pyf_sim_PPO_mask_TS-233472'


def prepare_base_folder(
    base_folder: str,
) -> None:
    base_path = common.prepare_save_paths(BASE_FOLDER, None, None, None)
    common.create_folder(base_path, delete_existing=OVERWRITE_FOLDERS)


def prepare_tb_path(
    base_folder: str,
    folder_tensorboard: str,
) -> Path:
    tensorboard_path = common.prepare_save_paths(base_folder, folder_tensorboard, None, None)
    common.create_folder(tensorboard_path, delete_existing=OVERWRITE_FOLDERS)
    # goto results folder, then execute
    # pdm run tensorboard --logdir="./tensorboard/PPO_mask_0/"
    return tensorboard_path


def prepare_model_path(
    base_folder: str,
) -> None:
    model_save_pth = common.prepare_save_paths(
        base_folder, FOLDER_MODEL_SAVEPOINTS, None, None
    )
    common.create_folder(model_save_pth, delete_existing=OVERWRITE_FOLDERS)


def get_save_path_model(
    num_timesteps: int,
    base_name: str = 'pyfsim_model',
) -> Path:
    filename: str = f'{base_name}_TS-{num_timesteps}'
    model_save_pth = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=filename,
        suffix='zip',
        include_timestamp=True,
    )

    return model_save_pth


def load_model() -> MaskablePPO:
    pth_model = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=FILENAME_TARGET_MODEL,
        suffix='zip',
    )
    model = MaskablePPO.load(pth_model)

    return model


def make_env(
    experiment_type: str,
    tensorboard_path: Path,
) -> Any:
    env = JSSEnv(experiment_type=experiment_type)
    check_env(env, warn=True)
    env = Monitor(env, filename=str(tensorboard_path), allow_early_resets=True)
    return env


def train(
    continue_learning: bool,
) -> None:
    prepare_base_folder(BASE_FOLDER)
    tensorboard_path = prepare_tb_path(BASE_FOLDER, FOLDER_TB)
    prepare_model_path(BASE_FOLDER)
    print(tensorboard_path)
    # StB automatically generates vector environment
    # vec_env = DummyVecEnv([make_env])
    env = make_env(EXP_TYPE, tensorboard_path)

    # model = PPO('MlpPolicy', vec_env, verbose=2, tensorboard_log=str(tensorboard_path))
    if continue_learning:
        model = load_model()
        model.set_env(env=env)
        calc_iterations = CALC_ITERATIONS
    else:
        model = MaskablePPO(
            'MlpPolicy', env, verbose=1, tensorboard_log=str(tensorboard_path)
        )
        calc_iterations = 0

    for it in range(1, (ITERATIONS + 1)):
        model.learn(
            total_timesteps=TIMESTEPS_PER_ITER,
            progress_bar=True,
            tb_log_name=f'{MODEL}',
            reset_num_timesteps=False,
        )
        num_timesteps: int = TIMESTEPS_PER_ITER * (it + calc_iterations)
        save_path_model = get_save_path_model(
            num_timesteps=num_timesteps, base_name=MODEL_BASE_NAME
        )
        if it % ITERATIONS_TILL_SAVE == 0:
            model.save(save_path_model)

    print('------------------')
    print('Training finished.')


def main() -> None:
    train(continue_learning=CONTINUE_LEARNING)


if __name__ == '__main__':
    main()
