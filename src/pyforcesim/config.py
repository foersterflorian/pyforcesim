from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Sequence
from datetime import datetime as Datetime
from pathlib import Path
from typing import Any, Final, Literal, cast, overload
from zoneinfo import ZoneInfo

from pyforcesim.types import (
    AgentDecisionTypes,
    Conf,
    ConfLib,
    ConfLibGymEnv,
    ConfLibGymEnvWIP,
    ConfLibGymEnvWIPTargets,
    ConfLibLogging,
    ConfLibSim,
    ConfLibSimSlack,
    ConfTensorboard,
    ConfTensorboardFiles,
    ConfTest,
    ConfTestFiles,
    ConfTestInputs,
    ConfTestRuns,
    ConfTrain,
    ConfTrainEnv,
    ConfTrainExperiment,
    ConfTrainFiles,
    ConfTrainModel,
    ConfTrainModelArch,
    ConfTrainModelInputs,
    ConfTrainModelSeeds,
    ConfTrainRuns,
    ConfTrainSB3,
    ConfTrainSystem,
    SB3ActorCriticNetworkArch,
    SB3PolicyArgs,
)

CFG_FILENAME: Final[str] = 'config.toml'
CFG_DEFAULT_PATH: Final[Path] = Path.cwd() / CFG_FILENAME


def get_cfg_path() -> Path:
    cfg_path_env = os.environ.get('PYFORCESIM_CFG_PATH', None)
    cfg_path: Path
    if cfg_path_env is not None:
        cfg_path = Path(cfg_path_env).absolute()
        if not cfg_path.exists():
            raise FileNotFoundError(
                'Config path via environment variable provided, '
                'but file does not seem to exist.'
            )
    else:
        if not CFG_DEFAULT_PATH.exists():
            raise FileNotFoundError(
                f'Config file not found under default location: >{CFG_DEFAULT_PATH}<'
            )
        cfg_path = CFG_DEFAULT_PATH

    return cfg_path


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, 'rb') as cfg_file:
        cfg = tomllib.load(cfg_file)

    print(f'Sucessfully loaded config from: {path}')

    return cfg


@overload
def _parse_int_sequence(
    seq: list[float] | float,
    param: str,
    mandatory: Literal[True],
) -> tuple[int, ...]: ...


@overload
def _parse_int_sequence(
    seq: list[float] | float,
    param: str,
    mandatory: Literal[False],
) -> tuple[int, ...] | None: ...


def _parse_int_sequence(
    seq: list[float] | float,
    param: str,
    mandatory: bool,
) -> tuple[int, ...] | None:
    parsed: tuple[int, ...] | None
    if not isinstance(seq, Sequence) and math.isnan(seq):
        parsed = None
    elif isinstance(seq, Sequence):
        parsed = tuple((int(seed) for seed in seq))
    else:
        raise ValueError(f'Could not parse values for >{param}< from config')

    if mandatory and parsed is None:
        raise ValueError(f'Parameter >{param}< was not set, but is mandatory')

    return parsed


def _parse_lib_cfg(cfg: dict[str, Any]) -> ConfLib:
    # lib.logging
    logging_enabled = cast(bool, cfg['lib']['logging']['enabled'])
    logging_to_file = cast(bool, cfg['lib']['logging']['file'])
    lib_logging = ConfLibLogging(enabled=logging_enabled, file=logging_to_file)
    # lib.sim.slack
    slack_init_as_upper_bound = cast(bool, cfg['lib']['sim']['slack']['init_as_upper_bound'])
    slack_use_threshold_upper = cast(bool, cfg['lib']['sim']['slack']['use_threshold_upper'])
    slack_overwrite_upper_bound = cast(
        float, cfg['lib']['sim']['slack']['overwrite_upper_bound']
    )
    default_lower_bound = cast(float, cfg['lib']['sim']['slack']['default_lower_bound'])
    slack_threshold_upper = cast(float, cfg['lib']['sim']['slack']['threshold_upper'])
    slack_min_range = cast(float, cfg['lib']['sim']['slack']['min_range'])
    slack_max_range = cast(float, cfg['lib']['sim']['slack']['max_range'])
    slack_adaption = cast(bool, cfg['lib']['sim']['slack']['adaption'])
    slack_adaption_min_upper_bound = cast(
        float, cfg['lib']['sim']['slack']['adaption_min_upper_bound']
    )
    slack_adaption_min_lower_bound = cast(
        float, cfg['lib']['sim']['slack']['adaption_min_lower_bound']
    )
    lib_sim_slack = ConfLibSimSlack(
        init_as_upper_bound=slack_init_as_upper_bound,
        use_threshold_upper=slack_use_threshold_upper,
        overwrite_upper_bound=slack_overwrite_upper_bound,
        default_lower_bound=default_lower_bound,
        threshold_upper=slack_threshold_upper,
        min_range=slack_min_range,
        max_range=slack_max_range,
        adaption=slack_adaption,
        adaption_min_upper_bound=slack_adaption_min_upper_bound,
        adaption_min_lower_bound=slack_adaption_min_lower_bound,
    )
    # lib.sim
    lib_sim = ConfLibSim(slack=lib_sim_slack)
    # lib.gym_env.WIP
    factor_WIP = cast(float, cfg['lib']['gym_env']['WIP']['factor_WIP'])
    if math.isnan(factor_WIP):
        factor_WIP = None
    else:
        factor_WIP = float(factor_WIP)
    use_WIP_targets = cast(bool, cfg['lib']['gym_env']['WIP']['use_WIP_targets'])
    WIP_relative_targets = cast(
        list[float], cfg['lib']['gym_env']['WIP']['WIP_relative_targets']
    )
    WIP_relative_targets = tuple(WIP_relative_targets)
    WIP_level_cycles = cast(int, cfg['lib']['gym_env']['WIP']['WIP_level_cycles'])
    WIP_relative_planned = cast(float, cfg['lib']['gym_env']['WIP']['WIP_relative_planned'])
    alpha = cast(float, cfg['lib']['gym_env']['WIP']['alpha'])
    lib_gym_WIP = ConfLibGymEnvWIP(
        factor_WIP=factor_WIP,
        use_WIP_targets=use_WIP_targets,
        WIP_relative_targets=WIP_relative_targets,
        WIP_level_cycles=WIP_level_cycles,
        WIP_relative_planned=WIP_relative_planned,
        alpha=alpha,
    )
    # lib.gym_env.WIP_targets
    min_ = cast(float, cfg['lib']['gym_env']['WIP_targets']['min'])
    max_ = cast(float, cfg['lib']['gym_env']['WIP_targets']['max'])
    number_WIP_levels = cast(int, cfg['lib']['gym_env']['WIP_targets']['number_WIP_levels'])
    lib_gym_WIP_targets = ConfLibGymEnvWIPTargets(
        min=min_, max=max_, number_WIP_levels=number_WIP_levels
    )
    # lib.gym_env
    sim_dur_weeks = cast(int, cfg['lib']['gym_env']['sim_dur_weeks'])
    buffer_size = cast(int, cfg['lib']['gym_env']['buffer_size'])
    job_pool_size_min = cast(int, cfg['lib']['gym_env']['job_pool_size_min'])
    job_pool_size_max = cast(int, cfg['lib']['gym_env']['job_pool_size_max'])
    dispatcher_seq_rule = cast(str, cfg['lib']['gym_env']['dispatcher_seq_rule'])
    dispatcher_alloc_rule = cast(str, cfg['lib']['gym_env']['dispatcher_alloc_rule'])
    lib_gym = ConfLibGymEnv(
        sim_dur_weeks=sim_dur_weeks,
        buffer_size=buffer_size,
        job_pool_size_min=job_pool_size_min,
        job_pool_size_max=job_pool_size_max,
        dispatcher_seq_rule=dispatcher_seq_rule,
        dispatcher_alloc_rule=dispatcher_alloc_rule,
        WIP=lib_gym_WIP,
        WIP_targets=lib_gym_WIP_targets,
    )
    # lib
    return ConfLib(logging=lib_logging, sim=lib_sim, gym_env=lib_gym)


def _parse_train_cfg(cfg: dict[str, Any]) -> ConfTrain:
    # train.system
    use_mp = cast(bool, cfg['train']['system']['multiprocessing'])
    n_procs = cast(float, cfg['train']['system']['number_processes'])
    if math.isnan(n_procs):
        n_procs = None
    else:
        n_procs = int(n_procs)
    save_state_actions = cast(bool, cfg['train']['system']['save_state_actions'])
    train_system = ConfTrainSystem(
        multiprocessing=use_mp,
        number_processes=n_procs,
        save_state_actions=save_state_actions,
    )
    # train.files
    overwrite_folders = cast(bool, cfg['train']['files']['overwrite_folders'])
    continue_learning = cast(bool, cfg['train']['files']['continue_learning'])
    folder_tensorboard = cast(str, cfg['train']['files']['folder_tensorboard'])
    folder_models = cast(str, cfg['train']['files']['folder_models'])
    model_name = cast(str, cfg['train']['files']['model_name'])
    filename_pretrained_model = cast(str, cfg['train']['files']['filename_pretrained_model'])
    train_files = ConfTrainFiles(
        overwrite_folders=overwrite_folders,
        continue_learning=continue_learning,
        folder_tensorboard=folder_tensorboard,
        folder_models=folder_models,
        model_name=model_name,
        filename_pretrained_model=filename_pretrained_model,
    )
    # train.experiment
    exp_number = cast(int, cfg['train']['experiment']['exp_number'])
    exp_number = str(exp_number)
    env_structure = cast(str, cfg['train']['experiment']['env_structure'])
    job_generation_method = cast(str, cfg['train']['experiment']['job_generation_method'])
    feedback_mechanism = cast(str, cfg['train']['experiment']['feedback_mechanism'])
    train_experiment = ConfTrainExperiment(
        exp_number=exp_number,
        env_structure=env_structure,
        job_generation_method=job_generation_method,
        feedback_mechanism=feedback_mechanism,
    )
    # train.model.inputs
    normalise_obs = cast(bool, cfg['train']['model']['inputs']['normalise_obs'])
    normalise_rew = cast(bool, cfg['train']['model']['inputs']['normalise_rew'])
    train_model_inputs = ConfTrainModelInputs(
        normalise_obs=normalise_obs, normalise_rew=normalise_rew
    )
    # train.model.seed
    rng = cast('list[float] | float', cfg['train']['model']['seeds']['rng'])
    rng = _parse_int_sequence(rng, param='train-rng', mandatory=False)
    eval = cast('list[float] | float', cfg['train']['model']['seeds']['eval'])
    eval = _parse_int_sequence(eval, param='train-eval', mandatory=True)
    train_model_seeds = ConfTrainModelSeeds(rng=rng, eval=eval)
    # train.model.arch
    batch_size = cast(int, cfg['train']['model']['arch']['batch_size'])
    sb3_arch = cast(SB3ActorCriticNetworkArch, cfg['train']['model']['arch']['sb3_arch'])
    train_model_arch = ConfTrainModelArch(batch_size=batch_size, sb3_arch=sb3_arch)
    # train.model
    train_model = ConfTrainModel(
        inputs=train_model_inputs, seeds=train_model_seeds, arch=train_model_arch
    )
    # train.runs
    ts_till_update = cast(int, cfg['train']['runs']['ts_till_update'])
    total_updates = cast(int, cfg['train']['runs']['total_updates'])
    updates_till_eval = cast(int, cfg['train']['runs']['updates_till_eval'])
    updates_till_savepoint = cast(int, cfg['train']['runs']['updates_till_savepoint'])
    num_eval_episodes = cast(int, cfg['train']['runs']['num_eval_episodes'])
    reward_threshold = cast(float, cfg['train']['runs']['reward_threshold'])
    if math.isnan(reward_threshold):
        reward_threshold = None
    else:
        reward_threshold = int(reward_threshold)
    train_runs = ConfTrainRuns(
        ts_till_update=ts_till_update,
        total_updates=total_updates,
        updates_till_eval=updates_till_eval,
        updates_till_savepoint=updates_till_savepoint,
        num_eval_episodes=num_eval_episodes,
        reward_threshold=reward_threshold,
    )
    # train.env
    randomise_reset = cast(bool, cfg['train']['env']['randomise_reset'])
    train_env = ConfTrainEnv(randomise_reset=randomise_reset)
    # train.sb3
    show_progressbar = cast(bool, cfg['train']['sb3']['show_progressbar'])
    train_sb3 = ConfTrainSB3(show_progressbar=show_progressbar)
    # train
    return ConfTrain(
        system=train_system,
        files=train_files,
        experiment=train_experiment,
        model=train_model,
        runs=train_runs,
        env=train_env,
        sb3=train_sb3,
    )


def _parse_test_cfg(cfg: dict[str, Any]) -> ConfTest:
    # test
    use_train_cfg = cast(bool, cfg['test']['use_train_config'])
    rng_seed = cast('list[float] | float', cfg['test']['seeds'])
    rng_seed = _parse_int_sequence(rng_seed, param='test-rng', mandatory=True)
    # test.files
    target_folder = cast(str, cfg['test']['files']['target_folder'])
    filename_target_model = cast(str, cfg['test']['files']['filename_target_model'])
    test_files = ConfTestFiles(
        target_folder=target_folder, filename_target_model=filename_target_model
    )
    # test.inputs
    normalise_obs = cast(bool, cfg['test']['inputs']['normalise_obs'])
    test_inputs = ConfTestInputs(normalise_obs=normalise_obs)
    # test.runs
    num_episodes = cast(int, cfg['test']['runs']['num_episodes'])
    perform_agent = cast(bool, cfg['test']['runs']['perform_agent'])
    perform_benchmark = cast(bool, cfg['test']['runs']['perform_benchmark'])
    test_runs = ConfTestRuns(
        num_episodes=num_episodes,
        perform_agent=perform_agent,
        perform_benchmark=perform_benchmark,
    )

    return ConfTest(
        use_train_config=use_train_cfg,
        seeds=rng_seed,
        files=test_files,
        inputs=test_inputs,
        runs=test_runs,
    )


def _parse_tensorboard_cfg(cfg: dict[str, Any]) -> ConfTensorboard:
    # tensorboard
    use_train_cfg = cast(bool, cfg['tensorboard']['use_train_config'])
    # tensorboard.files
    exp_folder = cast(str, cfg['tensorboard']['files']['exp_folder'])
    tb_files = ConfTensorboardFiles(exp_folder=exp_folder)

    return ConfTensorboard(use_train_config=use_train_cfg, files=tb_files)


def parse_cfg(cfg: dict[str, Any]) -> Conf:
    lib_cfg = _parse_lib_cfg(cfg=cfg)
    train_cfg = _parse_train_cfg(cfg=cfg)
    test_cfg = _parse_test_cfg(cfg=cfg)
    tb_cfg = _parse_tensorboard_cfg(cfg=cfg)

    return Conf(lib=lib_cfg, train=train_cfg, test=test_cfg, tensorboard=tb_cfg)


CFG_PATH: Final[Path] = get_cfg_path()
_cfg_raw = load_cfg(CFG_PATH)
CFG: Final[Conf] = parse_cfg(_cfg_raw)

# ** library: managed in constants

# ** training
# ** system
USE_MULTIPROCESSING: Final[bool] = CFG.train.system.multiprocessing
NUM_PROCS: Final[int | None] = CFG.train.system.number_processes
SAVE_STATE_ACTIONS: Final[bool] = CFG.train.system.save_state_actions
# ** files
OVERWRITE_FOLDERS: Final[bool] = CFG.train.files.overwrite_folders
CONTINUE_LEARNING: Final[bool] = CFG.train.files.continue_learning
FOLDER_TB: Final[str] = CFG.train.files.folder_tensorboard
FOLDER_MODEL_SAVEPOINTS: Final[str] = CFG.train.files.folder_models
MODEL: Final[str] = CFG.train.files.model_name
FILENAME_PRETRAINED_MODEL: Final[str] = CFG.train.files.filename_pretrained_model
MODEL_BASE_NAME: Final[str] = f'pyf_sim_{MODEL}'
# ** experiment characterisation
dt = Datetime.now(tz=ZoneInfo('Europe/Berlin')).replace(microsecond=0)
DATE: Final[str] = dt.strftime(r'%Y-%m-%d')
DEC_TYPE: Final[AgentDecisionTypes] = AgentDecisionTypes.SEQ
EXP_NUM: Final[str] = CFG.train.experiment.exp_number
ENV_STRUCTURE: Final[str] = CFG.train.experiment.env_structure
JOB_GEN_METHOD: Final[str] = CFG.train.experiment.job_generation_method
FEEDBACK_MACHANISM: Final[str] = CFG.train.experiment.feedback_mechanism
EXP_TYPE: Final[str] = f'{ENV_STRUCTURE}_{JOB_GEN_METHOD}'
EXPERIMENT_FOLDER: Final[str] = (
    f'{DATE}-{EXP_NUM.zfill(2)}__{ENV_STRUCTURE}__{JOB_GEN_METHOD}__{FEEDBACK_MACHANISM}'
)
BASE_FOLDER: Final[str] = f'results/{EXPERIMENT_FOLDER}'
# ** model input
NORMALISE_OBS: Final[bool] = CFG.train.model.inputs.normalise_obs
NORMALISE_REWARDS: Final[bool] = CFG.train.model.inputs.normalise_rew
# ** model seeding
RNG_SEEDS: Final[tuple[int, ...] | None] = CFG.train.model.seeds.rng
EVAL_SEEDS: Final[tuple[int, ...]] = CFG.train.model.seeds.eval
# ** model architecture
BATCH_SIZE: Final[int] = CFG.train.model.arch.batch_size
net_arch = CFG.train.model.arch.sb3_arch
POLICY_KWARGS: Final[SB3PolicyArgs] = {'net_arch': net_arch}
# ** runs
STEPS_TILL_UPDATE: Final[int] = CFG.train.runs.ts_till_update
NUM_EVAL_EPISODES: Final[int] = CFG.train.runs.num_eval_episodes
EVAL_FREQ: Final[int] = STEPS_TILL_UPDATE * CFG.train.runs.updates_till_eval
REWARD_THRESHOLD: Final[float | None] = CFG.train.runs.reward_threshold
TIMESTEPS_TOTAL: Final[int] = STEPS_TILL_UPDATE * CFG.train.runs.total_updates
STEPS_TILL_SAVE: Final[int] = STEPS_TILL_UPDATE * CFG.train.runs.updates_till_savepoint
# ** environment
RANDOMISE_RESET: Final[bool] = CFG.train.env.randomise_reset
# ** SB3 config
SHOW_PROGRESSBAR: Final[bool] = CFG.train.sb3.show_progressbar


# ** tests
TEST_USE_TRAIN_CONFIG: Final[bool] = CFG.test.use_train_config
TEST_RNG_SEEDS: Final[tuple[int, ...]] = CFG.test.seeds
# ** files
TEST_FILENAME_TARGET_MODEL: Final[str] = CFG.test.files.filename_target_model
TEST_TARGET_FOLDER: Final[str] = CFG.test.files.target_folder
# ** inputs
TEST_NORMALISE_OBS: Final[bool] = CFG.test.inputs.normalise_obs
# ** runs
TEST_NUM_EPISODES: Final[int] = CFG.test.runs.num_episodes
TEST_PERFORM_AGENT: Final[bool] = CFG.test.runs.perform_agent
TEST_PERFORM_BENCHMARK: Final[bool] = CFG.test.runs.perform_benchmark


# ** tensorboard
TB_USE_TRAIN_CONFIG: Final[bool] = CFG.tensorboard.use_train_config
# ** files
TB_EXP_FOLDER: Final[str] = CFG.tensorboard.files.exp_folder
