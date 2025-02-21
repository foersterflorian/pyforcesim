import math
import tomllib
from pathlib import Path
from typing import Any, Final, cast

from pyforcesim import common
from pyforcesim.types import (
    AgentDecisionTypes,
    Conf,
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


def load_cfg(path: Path) -> dict[str, Any]:
    assert path.exists(), 'config path not existing'

    with open(path, 'rb') as cfg_file:
        cfg = tomllib.load(cfg_file)

    return cfg


def _parse_train_cfg(cfg: dict[str, Any]) -> ConfTrain:
    # train.system
    use_mp = cast(bool, cfg['train']['system']['multiprocessing'])
    n_procs = cast(float, cfg['train']['system']['number_processes'])
    if math.isnan(n_procs):
        n_procs = None
    else:
        n_procs = int(n_procs)
    train_system = ConfTrainSystem(multiprocessing=use_mp, number_processes=n_procs)
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
    rng = cast(float, cfg['train']['model']['seeds']['rng'])
    if math.isnan(rng):
        rng = None
    else:
        rng = int(rng)
    eval = int(cfg['train']['model']['seeds']['eval'])
    train_model_seeds = ConfTrainModelSeeds(rng=rng, eval=eval)
    # train.model.arch
    sb3_arch = cast(SB3ActorCriticNetworkArch, cfg['train']['model']['arch']['sb3_arch'])
    train_model_arch = ConfTrainModelArch(sb3_arch=sb3_arch)
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
    rng_seed = cast(int, cfg['test']['seed'])
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
        seed=rng_seed,
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
    train_cfg = _parse_train_cfg(cfg=cfg)
    test_cfg = _parse_test_cfg(cfg=cfg)
    tb_cfg = _parse_tensorboard_cfg(cfg=cfg)

    return Conf(train=train_cfg, test=test_cfg, tensorboard=tb_cfg)


CFG_FILENAME: Final[str] = 'config.toml'
CFG_PATH: Final[Path] = Path.cwd() / CFG_FILENAME
cfg_raw = load_cfg(CFG_PATH)
CFG: Final[Conf] = parse_cfg(cfg_raw)

# ** training
# ** system
USE_MULTIPROCESSING: Final[bool] = CFG.train.system.multiprocessing
NUM_PROCS: Final[int | None] = CFG.train.system.number_processes
# ** files
OVERWRITE_FOLDERS: Final[bool] = CFG.train.files.overwrite_folders
CONTINUE_LEARNING: Final[bool] = CFG.train.files.continue_learning
FOLDER_TB: Final[str] = CFG.train.files.folder_tensorboard
FOLDER_MODEL_SAVEPOINTS: Final[str] = CFG.train.files.folder_models
MODEL: Final[str] = CFG.train.files.model_name
FILENAME_PRETRAINED_MODEL: Final[str] = CFG.train.files.filename_pretrained_model
MODEL_BASE_NAME: Final[str] = f'pyf_sim_{MODEL}'
# ** experiment characterisation
DATE = common.get_timestamp(with_time=False)
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
RNG_SEED: Final[int | None] = CFG.train.model.seeds.rng
EVAL_SEED: Final[int] = CFG.train.model.seeds.eval
# ** model architecture
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
TEST_RNG_SEED: Final[int] = CFG.test.seed
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
