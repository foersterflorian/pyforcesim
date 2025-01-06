import os
import warnings
from pathlib import Path
from typing import Any, Final

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pyforcesim import common
from pyforcesim.rl.gym_env import JSSEnv
from pyforcesim.rl.sb3.custom_callbacks import MaskableEvalCallback as EvalCallback
from pyforcesim.types import AgentDecisionTypes, BuilderFuncFamilies

# ** input
OVERWRITE_FOLDERS: Final[bool] = True
CONTINUE_LEARNING: Final[bool] = False
NORMALISE_OBS: Final[bool] = True
CALC_ITERATIONS: Final[int] = 69632 // 2048
RNG_SEED: Final[int] = 42

DATE = common.get_timestamp(with_time=False)
DEC_TYPE: Final[AgentDecisionTypes] = AgentDecisionTypes.SEQ
EXP_NUM: Final[str] = '1'
ENV_STRUCTURE: Final[str] = '1-3-7'
JOB_GEN_METHOD: Final[str] = 'ConstIdeal'
EXP_TYPE: Final[str] = f'{ENV_STRUCTURE}_{JOB_GEN_METHOD}'
FEEDBACK_MACHANISM: Final[str] = 'Util'
EXPERIMENT_FOLDER: Final[str] = (
    f'{DATE}-{EXP_NUM.zfill(2)}__{ENV_STRUCTURE}__{JOB_GEN_METHOD}__{FEEDBACK_MACHANISM}'
)
BASE_FOLDER: Final[str] = f'results/{EXPERIMENT_FOLDER}'

FOLDER_TB: Final[str] = 'tensorboard'
FOLDER_MODEL_SAVEPOINTS: Final[str] = 'models'

MODEL: Final[str] = 'PPO_mask'
MODEL_BASE_NAME: Final[str] = f'pyf_sim_{MODEL}'
NUM_EVAL_EPISODES: Final[int] = 2
EVAL_FREQ: Final[int] = 2048 * 2
REWARD_THRESHOLD: Final[float | None] = None  # -0.01
TIMESTEPS_PER_ITER: Final[int] = int(2048 * 1)
ITERATIONS: Final[int] = 20
ITERATIONS_TILL_SAVE: Final[int] = 2

FILENAME_PRETRAINED_MODEL: Final[str] = '2024-07-23--16-20-52_pyf_sim_PPO_mask_TS-69632'


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
    seed: int | None = None,
    verify_env: bool = True,
    sim_randomise_reset: bool = False,
) -> Any:
    sim_check_agent_feasibility: bool = True
    if verify_env:
        sim_check_agent_feasibility = False

    env = JSSEnv(
        experiment_type=experiment_type,
        agent_type=DEC_TYPE,
        gantt_chart_on_termination=gantt_chart,
        seed=seed,
        sim_randomise_reset=sim_randomise_reset,
        sim_check_agent_feasibility=sim_check_agent_feasibility,
        builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
    )

    if verify_env:
        check_env(env, warn=True)
        # recreate to ensure that nothing was altered
        env = JSSEnv(
            experiment_type=experiment_type,
            agent_type=DEC_TYPE,
            gantt_chart_on_termination=gantt_chart,
            seed=seed,
            sim_randomise_reset=sim_randomise_reset,
            sim_check_agent_feasibility=sim_check_agent_feasibility,
            builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
        )
    tb_path: str | None = None
    if tensorboard_path is not None:
        tb_path = str(tensorboard_path)
    env = Monitor(env, filename=tb_path, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])  # type: ignore
    env.seed(seed=seed)
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

    env = make_env(
        EXP_TYPE,
        tensorboard_path,
        normalise_obs=NORMALISE_OBS,
        seed=RNG_SEED,
    )
    eval_env = make_env(
        EXP_TYPE,
        tensorboard_path,
        normalise_obs=NORMALISE_OBS,
        seed=RNG_SEED,
    )
    # ** reward threshold callback
    reward_thresh_callback: StopTrainingOnRewardThreshold | None = None
    if REWARD_THRESHOLD is not None:
        reward_thresh_callback = StopTrainingOnRewardThreshold(
            reward_threshold=REWARD_THRESHOLD,
            verbose=1,
        )

    best_model_save_pth = common.prepare_save_paths(
        base_folder=BASE_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=None,
        suffix=None,
        include_timestamp=False,
    )
    eval_callback = EvalCallback(
        callback_on_new_best=reward_thresh_callback,
        eval_env=eval_env,
        best_model_save_path=str(best_model_save_pth),
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        use_masking=True,
        verbose=1,
    )

    if continue_learning:
        pth_vec_norm, model = load_model()
        model.set_env(env=env)
        calc_iterations = CALC_ITERATIONS
        if NORMALISE_OBS:
            if not pth_vec_norm.exists():
                raise FileNotFoundError(f'VecNormalize info not found under: {pth_vec_norm}')
            env = VecNormalize.load(str(pth_vec_norm), env)
            env.training = True
            print('Normalization info loaded successfully.')
    else:
        model = MaskablePPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_path),
            seed=RNG_SEED,
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
        num_timesteps = TIMESTEPS_PER_ITER * (it + calc_iterations)
        save_path_model, save_path_vec_norm = get_save_path_model(
            num_timesteps=num_timesteps, base_name=MODEL_BASE_NAME
        )
        # break early if training should not be continued
        if not eval_callback.continue_training:
            # rename best model and vec normalize info if present
            best_model_file = best_model_save_pth.joinpath('best_model.zip')
            os.rename(best_model_file, save_path_model)
            best_model_vec_norm: Path | None = None
            vec_norm = model.get_vec_normalize_env()
            if vec_norm is not None:
                best_model_vec_norm = best_model_save_pth.joinpath('best_model_vec_norm.pkl')
                os.rename(best_model_vec_norm, save_path_vec_norm)

            break
        # save regularly following config
        if it % ITERATIONS_TILL_SAVE == 0:
            model.save(save_path_model)
            vec_norm = model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(str(save_path_vec_norm))

    # rename best model and vec normalize info if present
    save_path_model, save_path_vec_norm = get_save_path_model(
        num_timesteps=(num_timesteps + 1), base_name=MODEL_BASE_NAME
    )
    best_model_file = best_model_save_pth.joinpath('best_model.zip')
    os.rename(best_model_file, save_path_model)

    best_model_vec_norm: Path | None = None
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        best_model_vec_norm = best_model_save_pth.joinpath('best_model_vec_norm.pkl')
        os.rename(best_model_vec_norm, save_path_vec_norm)

    print('------------------')
    print('Training finished.')


def main() -> None:
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message=r'^[\s]*.*to get variables from other wrappers is deprecated.*$',
    )
    train(continue_learning=CONTINUE_LEARNING)


if __name__ == '__main__':
    main()
