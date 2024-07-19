from pathlib import Path
from typing import Any, Final

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pyforcesim import common
from pyforcesim import datetime as pyf_dt
from pyforcesim.rl.gym_env import JSSEnv

# ** input
OVERWRITE_FOLDERS: Final[bool] = True
CONTINUE_LEARNING: Final[bool] = False
NORMALISE_OBS: Final[bool] = True
CALC_ITERATIONS: Final[int] = 233472 // 2048

DATE = common.get_timestamp(with_time=False)
EXP_NUM: Final[str] = '03'
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
NUM_EVAL_EPISODES: Final[int] = 3
EVAL_FREQ: Final[int] = 2048 * 4
REWARD_THREHSHOLD: Final[float] = -0.1
TIMESTEPS_PER_ITER: Final[int] = int(2048 * 1)
ITERATIONS: Final[int] = 50
ITERATIONS_TILL_SAVE: Final[int] = 5

FILENAME_PRETRAINED_MODEL: Final[str] = '2024-06-24--19-35-38_pyf_sim_PPO_mask_TS-233472'


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
) -> tuple[Path, Path]:
    filename: str = f'{base_name}_TS-{num_timesteps}'
    model_save_pth = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=filename,
        suffix='zip',
        include_timestamp=True,
    )
    vec_norm_filename = f'{filename}_vec_norm'
    vec_norm_save_pth = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=vec_norm_filename,
        suffix='pkl',
        include_timestamp=True,
    )

    return model_save_pth, vec_norm_save_pth


def load_model() -> tuple[Path, MaskablePPO]:
    pth_model = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=FILENAME_PRETRAINED_MODEL,
        suffix='zip',
    )
    vec_norm_filename = f'{FILENAME_PRETRAINED_MODEL}_vec_norm'
    pth_vec_norm = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=vec_norm_filename,
        suffix='pkl',
    )
    model = MaskablePPO.load(pth_model)

    return pth_vec_norm, model


def make_env(
    experiment_type: str,
    tensorboard_path: Path | None,
    gantt_chart: bool = False,
    normalise_obs: bool = True,
) -> Any:
    env = JSSEnv(
        experiment_type=experiment_type,
        gantt_chart_on_termination=gantt_chart,
    )
    check_env(env, warn=True)
    tb_path: str | None = None
    if tensorboard_path is not None:
        tb_path = str(tensorboard_path)
    env = Monitor(env, filename=tb_path, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])  # type: ignore
    if normalise_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return env


def train(
    continue_learning: bool,
) -> None:
    prepare_base_folder(BASE_FOLDER)
    tensorboard_path = prepare_tb_path(BASE_FOLDER, FOLDER_TB)
    prepare_model_path(BASE_FOLDER)
    tensorboard_command = f'pdm run tensorboard --logdir="{tensorboard_path}"'
    print('tensorboard command: ', tensorboard_command)
    # StB automatically generates vector environment
    # vec_env = DummyVecEnv([make_env])
    # gym_env = make_env(EXP_TYPE, tensorboard_path)
    env = make_env(EXP_TYPE, tensorboard_path, normalise_obs=NORMALISE_OBS)
    eval_env = make_env(EXP_TYPE, tensorboard_path, normalise_obs=NORMALISE_OBS)
    reward_thresh_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THREHSHOLD,
    )

    best_model_save_pth = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=None,
        suffix=None,
        include_timestamp=False,
    )
    eval_callback = EvalCallback(
        callback_on_new_best=None,
        eval_env=eval_env,
        best_model_save_path=str(best_model_save_pth),
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        use_masking=True,
    )

    if continue_learning:
        pth_vec_norm, model = load_model()
        model.set_env(env=env)
        calc_iterations = CALC_ITERATIONS
        if not pth_vec_norm.exists():
            raise FileNotFoundError(f'VecNormalize info not found under: {pth_vec_norm}')
        env = VecNormalize.load(str(pth_vec_norm), env)
        env.training = False
        print('Normalization info loaded successfully.')
    else:
        model = MaskablePPO(
            'MlpPolicy', env, verbose=1, tensorboard_log=str(tensorboard_path)
        )
        calc_iterations = 0

    for it in range(1, (ITERATIONS + 1)):
        model.learn(
            callback=eval_callback,
            total_timesteps=TIMESTEPS_PER_ITER,
            progress_bar=True,
            tb_log_name=f'{MODEL}',
            reset_num_timesteps=False,
        )
        num_timesteps: int = TIMESTEPS_PER_ITER * (it + calc_iterations)
        save_path_model, save_path_vec_norm = get_save_path_model(
            num_timesteps=num_timesteps, base_name=MODEL_BASE_NAME
        )
        if it % ITERATIONS_TILL_SAVE == 0:
            model.save(save_path_model)
            vec_norm = model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(str(save_path_vec_norm))

    print('------------------')
    print('Training finished.')


def main() -> None:
    train(continue_learning=CONTINUE_LEARNING)


if __name__ == '__main__':
    main()
