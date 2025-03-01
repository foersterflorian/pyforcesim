import os
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import psutil
import stable_baselines3.common.vec_env.subproc_vec_env as sb3_to_patch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import sb3_monkeypatch
from pyforcesim import common
from pyforcesim.config import (
    BASE_FOLDER,
    CONTINUE_LEARNING,
    DEC_TYPE,
    EVAL_FREQ,
    EVAL_SEEDS,
    EXP_TYPE,
    FILENAME_PRETRAINED_MODEL,
    FOLDER_MODEL_SAVEPOINTS,
    FOLDER_TB,
    MODEL,
    MODEL_BASE_NAME,
    NORMALISE_OBS,
    NORMALISE_REWARDS,
    NUM_EVAL_EPISODES,
    NUM_PROCS,
    OVERWRITE_FOLDERS,
    POLICY_KWARGS,
    RANDOMISE_RESET,
    REWARD_THRESHOLD,
    RNG_SEEDS,
    SHOW_PROGRESSBAR,
    STEPS_TILL_SAVE,
    STEPS_TILL_UPDATE,
    TIMESTEPS_TOTAL,
    USE_MULTIPROCESSING,
)
from pyforcesim.rl.gym_env import JSSEnv
from pyforcesim.rl.sb3.custom_callbacks import MaskableEvalCallback as EvalCallback
from pyforcesim.types import (
    BuilderFuncFamilies,
)

# ** monkeypatch SB3 to work with SubprocVecEnv and MaskablePPO from sb3_contrib
sb3_to_patch._worker = sb3_monkeypatch.worker


def prepare_base_folder(
    base_folder: str,
    overwrite_existing: bool,
) -> Path:
    base_path = common.prepare_save_paths(base_folder, None, None, None)
    common.create_folder(base_path, delete_existing=overwrite_existing)

    return base_path


def prepare_tb_path(
    base_folder: str,
    folder_tensorboard: str,
    overwrite_existing: bool,
) -> Path:
    tensorboard_path = common.prepare_save_paths(base_folder, folder_tensorboard, None, None)
    common.create_folder(tensorboard_path, delete_existing=overwrite_existing)
    return tensorboard_path


def prepare_model_path(
    base_folder: str,
    model_folder_name: str,
    overwrite_existing: bool,
) -> Path:
    model_save_pth = common.prepare_save_paths(base_folder, model_folder_name, None, None)
    common.create_folder(model_save_pth, delete_existing=overwrite_existing)

    return model_save_pth


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
    model = MaskablePPO.load(pth_model, device='cpu')

    return pth_vec_norm, model


def _get_num_cpu_cores(
    num_procs: int | None,
) -> int:
    sys_cores = psutil.cpu_count(logical=False)
    assert sys_cores is not None, 'system CPU count not available'
    max_cores = sys_cores - 1

    if num_procs is None or num_procs > max_cores:
        return max_cores
    else:
        return num_procs


def get_relevant_seed(
    seeds: Sequence[int] | None,
) -> int | None:
    seed: int | None = None
    if seeds is not None:
        seed = seeds[0]  # use first seed as initial

    return seed


def make_subproc_env(
    experiment_type: str,
    tensorboard_path: Path | None,
    num_procs: int,
    gantt_chart: bool = False,
    normalise_obs: bool = True,
    normalise_rewards: bool = False,
    seeds: Sequence[int] | None = None,
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
        seeds=seeds,
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
            seeds=seeds,
            sim_randomise_reset=sim_randomise_reset,
            sim_check_agent_feasibility=sim_check_agent_feasibility,
            builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
        )

    def make_multi_env(ident: int, change_seeds: bool) -> Callable[[], Monitor]:
        seed_per_env: int | None = None
        if seeds is not None:
            seed_per_env = seeds[0]

        training_seeds: tuple[int, ...] | None = None
        if seed_per_env is not None and change_seeds:
            seed_per_env += ident
            training_seeds = (seed_per_env,)

        def _initialise_env() -> Monitor:
            env = JSSEnv(
                experiment_type=experiment_type,
                agent_type=DEC_TYPE,
                gantt_chart_on_termination=gantt_chart,
                seeds=training_seeds,
                sim_randomise_reset=sim_randomise_reset,
                sim_check_agent_feasibility=sim_check_agent_feasibility,
                builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
            )
            # each process a monitor file
            if tensorboard_path is not None:
                tb_path_proc = tensorboard_path / str(ident)
            env = Monitor(env, filename=str(tb_path_proc), allow_early_resets=True)
            return env

        return _initialise_env

    env = SubprocVecEnv(
        [make_multi_env(ident, change_seeds=False) for ident in range(num_procs)],
        start_method='spawn',
    )  # type: ignore

    seed = get_relevant_seed(seeds)
    env.seed(seed=seed)
    if normalise_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=normalise_rewards, clip_obs=10.0)

    return env


def make_env(
    experiment_type: str,
    tensorboard_path: Path | None,
    gantt_chart: bool = False,
    normalise_obs: bool = True,
    normalise_rewards: bool = False,
    seeds: Sequence[int] | None = None,
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
        seeds=seeds,
        sim_randomise_reset=sim_randomise_reset,
        sim_check_agent_feasibility=sim_check_agent_feasibility,
        builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
    )

    if verify_env:
        check_env(env, warn=True)
        # recreate to ensure that nothing was altered
        del env
        env = JSSEnv(
            experiment_type=experiment_type,
            agent_type=DEC_TYPE,
            gantt_chart_on_termination=gantt_chart,
            seeds=seeds,
            sim_randomise_reset=sim_randomise_reset,
            sim_check_agent_feasibility=sim_check_agent_feasibility,
            builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
        )
    tb_path: str | None = None
    if tensorboard_path is not None:
        tb_path = str(tensorboard_path)
    env = Monitor(env, filename=tb_path, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])  # type: ignore

    seed = get_relevant_seed(seeds)
    env.seed(seed=seed)
    if normalise_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=normalise_rewards, clip_obs=10.0)

    return env


def train(
    continue_learning: bool,
    seeds: Sequence[int] | None,
    eval_seeds: Sequence[int],
    normalise_rewards: bool,
    sim_randomise_reset: bool = False,
    use_mp: bool = False,
    num_procs: int | None = None,
) -> None:
    _ = prepare_base_folder(BASE_FOLDER, OVERWRITE_FOLDERS)
    tensorboard_path = prepare_tb_path(BASE_FOLDER, FOLDER_TB, OVERWRITE_FOLDERS)
    model_folder = prepare_model_path(BASE_FOLDER, FOLDER_MODEL_SAVEPOINTS, OVERWRITE_FOLDERS)
    tensorboard_command = f'pdm run tensorboard --logdir="{tensorboard_path}"'
    print('tensorboard command: ', tensorboard_command)

    if use_mp:
        num_procs = _get_num_cpu_cores(num_procs)
        env = make_subproc_env(
            EXP_TYPE,
            tensorboard_path,
            num_procs=num_procs,
            normalise_obs=NORMALISE_OBS,
            normalise_rewards=normalise_rewards,
            seeds=seeds,
            sim_randomise_reset=sim_randomise_reset,
        )
    else:
        env = make_env(
            EXP_TYPE,
            tensorboard_path,
            normalise_obs=NORMALISE_OBS,
            normalise_rewards=normalise_rewards,
            seeds=seeds,
            sim_randomise_reset=sim_randomise_reset,
        )
    # ** Checkpoint
    n_envs: int = 1
    if num_procs is not None:
        n_envs = num_procs
    save_freq = max(STEPS_TILL_SAVE // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_folder),
        name_prefix=MODEL_BASE_NAME,
        save_vecnormalize=NORMALISE_OBS,
        verbose=2,
    )
    # ** evaluation
    eval_env = make_env(
        EXP_TYPE,
        tensorboard_path,
        normalise_obs=NORMALISE_OBS,
        normalise_rewards=False,
        seeds=eval_seeds,
    )
    # ** reward threshold callback
    reward_thresh_callback: StopTrainingOnRewardThreshold | None = None
    if REWARD_THRESHOLD is not None:
        reward_thresh_callback = StopTrainingOnRewardThreshold(
            reward_threshold=REWARD_THRESHOLD,
            verbose=1,
        )
    eval_callback = EvalCallback(
        callback_on_new_best=reward_thresh_callback,
        eval_env=eval_env,
        best_model_save_path=str(model_folder),
        n_eval_episodes=len(eval_seeds),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        use_masking=True,
        verbose=1,
    )

    if continue_learning:
        pth_vec_norm, model = load_model()
        model.set_env(env=env)
        if NORMALISE_OBS and not pth_vec_norm.exists():
            raise FileNotFoundError(f'VecNormalize info not found under: {pth_vec_norm}')
        elif NORMALISE_OBS:
            env = VecNormalize.load(str(pth_vec_norm), env)
            env.training = True
            print('Normalization info loaded successfully.')
    else:
        seed = get_relevant_seed(seeds)
        model = MaskablePPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_path),
            seed=seed,
            n_steps=STEPS_TILL_UPDATE,
            policy_kwargs=POLICY_KWARGS,  # type: ignore
            device='cpu',
        )

    print('=============================================================')
    print(f'Network architecture is: {model.policy}')
    print('=============================================================')
    model.learn(
        callback=[eval_callback, checkpoint_callback],
        total_timesteps=TIMESTEPS_TOTAL,
        progress_bar=SHOW_PROGRESSBAR,
        tb_log_name=f'{MODEL}',
        reset_num_timesteps=False,
    )

    print('------------------')
    print('Training finished.')


def main() -> None:
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message=r'^[\s]*.*to get variables from other wrappers is deprecated.*$',
    )
    train(
        continue_learning=CONTINUE_LEARNING,
        seeds=RNG_SEEDS,
        eval_seeds=EVAL_SEEDS,
        normalise_rewards=NORMALISE_REWARDS,
        sim_randomise_reset=RANDOMISE_RESET,
        use_mp=USE_MULTIPROCESSING,
        num_procs=NUM_PROCS,
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Renaming best model if available...')
    except Exception as err:
        print('Following exception occured: ', err)
    finally:
        time.sleep(5)  # wait for other processes to end (for SubprocVecEnv)
        # rename best model and vec normalize info if present
        model_folder = prepare_model_path(BASE_FOLDER, FOLDER_MODEL_SAVEPOINTS, False)
        save_path_model, save_path_vec_norm = get_save_path_model(
            num_timesteps=(TIMESTEPS_TOTAL + 1), base_name=MODEL_BASE_NAME
        )
        best_model_file = model_folder / 'best_model.zip'
        if best_model_file.exists():
            os.rename(best_model_file, save_path_model)

        best_model_vec_norm_file = model_folder / 'best_model_vec_norm.pkl'
        if best_model_vec_norm_file.exists():
            os.rename(best_model_vec_norm_file, save_path_vec_norm)

        print('Best model renamed successfully')
