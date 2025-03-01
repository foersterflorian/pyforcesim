from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pyforcesim import common, loggers
from pyforcesim import datetime as pyf_dt
from pyforcesim.config import (
    BASE_FOLDER,
    DEC_TYPE,
    EXP_TYPE,
    FOLDER_MODEL_SAVEPOINTS,
    RNG_SEEDS,
    TEST_FILENAME_TARGET_MODEL,
    TEST_NORMALISE_OBS,
    TEST_NUM_EPISODES,
    TEST_PERFORM_AGENT,
    TEST_PERFORM_BENCHMARK,
    TEST_RNG_SEEDS,
    TEST_TARGET_FOLDER,
    TEST_USE_TRAIN_CONFIG,
)
from pyforcesim.constants import TimeUnitsTimedelta
from pyforcesim.rl.gym_env import JSSEnv
from pyforcesim.types import BuilderFuncFamilies, TestResults
from train import get_relevant_seed, make_env

if TYPE_CHECKING:
    from pyforcesim.rl.agents import ValidateAllocationAgent, ValidateSequencingAgent


# ** files
model_properties_pattern = re.compile(r'(?:pyf_sim_)([\w]+)_(TS-[\d]+)$')
matches = model_properties_pattern.search(TEST_FILENAME_TARGET_MODEL)
if matches is None:
    raise ValueError(
        f'Model properties could not be extracted out of: {TEST_FILENAME_TARGET_MODEL}'
    )
ALGO_TYPE: Final[str] = matches.group(1)
TIMESTEPS: Final[str] = matches.group(2)

USER_FOLDER: Final[str] = f'results/{TEST_TARGET_FOLDER}'
user_exp_type_pattern = re.compile(
    r'^([\d\-]*)(?:[_]*)([\d\-]*)(?:[_]*)([a-zA-Z]*)(?:[_]*)([a-zA-Z]*)$'
)
matches = user_exp_type_pattern.match(TEST_TARGET_FOLDER)
if matches is None:
    raise ValueError(f'Experiment type could not be extracted out of: {TEST_TARGET_FOLDER}')

USER_EXP_TYPE: Final[str] = f'{matches.group(2)}_{matches.group(3)}'
USER_RNG_SEEDS: Final[tuple[int, ...]] = TEST_RNG_SEEDS

ROOT_FOLDER = USER_FOLDER
ROOT_EXP_TYPE = USER_EXP_TYPE
ROOT_RNG_SEEDS = USER_RNG_SEEDS
# if TEST_USE_TRAIN_CONFIG and RNG_SEEDS is None:
#     raise ValueError(
#         'Eval seeds can not be None. Do not use train config if it contains no seeds'
#     )
# elif TEST_USE_TRAIN_CONFIG:
#     ROOT_FOLDER = BASE_FOLDER
#     ROOT_EXP_TYPE = EXP_TYPE
#     ROOT_RNG_SEEDS = RNG_SEEDS

VAL_EXP_TYPE: Final[str] = f'{ROOT_EXP_TYPE}_validate'
VAL_EXP_EPISODES: Final[int] = 1


def load_model(env: VecNormalize | JSSEnv | None = None) -> MaskablePPO:
    # ** load model
    pth_model = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=TEST_FILENAME_TARGET_MODEL,
        suffix='zip',
    )

    model = MaskablePPO.load(pth_model, env)

    return model


def get_path_vec_norm() -> Path:
    vec_norm_filename = f'{TEST_FILENAME_TARGET_MODEL}_vec_norm'
    pth_vec_norm = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=FOLDER_MODEL_SAVEPOINTS,
        filename=vec_norm_filename,
        suffix='pkl',
    )
    return pth_vec_norm


def export_gantt_chart(
    env: JSSEnv,
    is_benchmark: bool,
    episode_num: int,
    cum_reward: float,
    auto_open_html: bool = False,
) -> None:
    # use last seed: at this point VecEnv already reset, thus new seed is
    # set for the next run
    seed = env.last_seed if env.last_seed is not None else 'n.a.'
    gantt_chart = env.last_gantt_chart
    # ** KPIs
    if gantt_chart is None:
        raise ValueError('Gantt chart is >>None<<')

    cycle_time = env.cycle_time
    if cycle_time is None:
        raise ValueError('Cycle time is >>None<<')

    mean_utilisation = env.sim_utilisation
    if mean_utilisation is None:
        raise ValueError('Mean utilisation is >>None<<')

    end_dev_mean = env.end_date_dev_mean
    end_dev_std = env.end_date_dev_std
    if end_dev_mean is None or end_dev_std is None:
        raise ValueError(
            'One statistical metric for the ending date deviation not available.'
        )
    norm_td = pyf_dt.timedelta_from_val(1, TimeUnitsTimedelta.HOURS)
    end_dev_mean_hours = end_dev_mean / norm_td
    end_dev_std_hours = end_dev_std / norm_td

    waiting_chosen = env.waiting_chosen
    jobs_total = env.jobs_total
    jobs_tardy = env.jobs_tardy
    jobs_tardy_perc = jobs_tardy / jobs_total
    jobs_early = env.jobs_early
    jobs_early_perc = jobs_early / jobs_total
    jobs_punctual = env.jobs_punctual
    jobs_punctual_perc = jobs_punctual / jobs_total

    title_KPIs = (
        f'seed: {seed}, number of waiting actions: {waiting_chosen}<br>'
        f'cycle time: {cycle_time}, mean util: {mean_utilisation:.6%}'
        f'<br>ending date deviation: '
        f'mean: {end_dev_mean_hours:.6f}, std: {end_dev_std_hours:.6f}<br>'
        f'jobs: total={jobs_total}, punctual={jobs_punctual} ({jobs_punctual_perc:.2%}), '
        f'early={jobs_early} ({jobs_early_perc:.2%}), '
        f'tardy={jobs_tardy} ({jobs_tardy_perc:.2%})'
    )
    title_reward = f'<br>Episode: {episode_num}, Cum Reward: {cum_reward:.4f}'

    if is_benchmark:
        policy_name = env.policy_name if env.policy_name is not None else 'n.a.'
        title = (
            f'Gantt Chart<br>Benchmark (Policy: {policy_name})<br>ExpType: {VAL_EXP_TYPE}, '
        )
        title_chart = title + title_KPIs + title_reward
        filename = f'Benchmark_{policy_name}_Episode_{episode_num}_Seed_{seed}'
    else:
        title = (
            f'Gantt Chart<br>Model (Algo: {ALGO_TYPE}, Timesteps: '
            f'{TIMESTEPS})<br>ExpType: {ROOT_EXP_TYPE}, '
        )
        title_chart = title + title_KPIs + title_reward
        filename = f'{ALGO_TYPE}_{TIMESTEPS}_Episode_{episode_num}_Seed_{seed}'

    gantt_chart.update_layout(title=title_chart, margin=dict(t=275))
    save_pth = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=None,
        filename=filename,
        suffix='html',
    )
    gantt_chart.write_html(save_pth, auto_open=auto_open_html)


def export_dbs(
    env: JSSEnv,
    is_benchmark: bool,
    episode_num: int,
) -> None:
    # use last seed: at this point VecEnv already reset, thus new seed is
    # set for the next run
    seed = env.last_seed if env.last_seed is not None else 'n.a.'
    filename_base = f'{ALGO_TYPE}_{TIMESTEPS}_Episode_{episode_num}_Seed_{seed}'
    filename_job_db = f'{filename_base}-job-db'
    filename_op_db = f'{filename_base}-op-db'
    if is_benchmark:
        policy_name = env.policy_name if env.policy_name is not None else 'n.a.'
        filename_job_db = f'Benchmark_{policy_name}_Episode_{episode_num}_job-db_Seed_{seed}'
        filename_op_db = f'Benchmark_{policy_name}_Episode_{episode_num}_op-db_Seed_{seed}'

    save_pth_job_db = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=None,
        filename=filename_job_db,
        suffix='pkl',
    )
    save_pth_op_db = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=None,
        filename=filename_op_db,
        suffix='pkl',
    )
    job_db = env.last_job_db
    op_db = env.last_op_db
    if job_db is not None and op_db is not None:
        job_db.to_pickle(save_pth_job_db)
        op_db.to_pickle(save_pth_op_db)
    else:
        raise ValueError('Either job DB or operation DB >>None<<')


def export_results(
    results: TestResults,
    filename: str,
) -> None:
    save_pth = common.prepare_save_paths(
        base_folder=ROOT_FOLDER,
        target_folder=None,
        filename=filename,
        suffix='json',
    )

    with open(save_pth, 'w') as file:
        json.dump(results, file)

    print(f'Saved results successfully under: >{save_pth}<')


def callback(locals, *_):
    done = cast(bool, locals['done'])

    if not done:
        return

    vec_env = cast(DummyVecEnv, locals['env'])
    env = cast('Monitor | JSSEnv', vec_env.envs[0])
    if isinstance(env, Monitor):
        env = cast(JSSEnv, env.env)

    cum_rewards = cast(npt.NDArray[np.float32], locals['current_rewards'])
    cum_reward = cum_rewards[0]
    episode_counters = cast(npt.NDArray[np.int32], locals['episode_counts'])
    episode_num = episode_counters[0]
    print(f'[AGENT EVAL] Episode {episode_num + 1}: Reward = {cum_reward:.4f}')
    print(
        (
            f'[AGENT EVAL] Episode {episode_num + 1}: Cycle Time = {env.cycle_time}, '
            f'Utilisation = {env.sim_utilisation:.4%}'
        )
    )

    export_gantt_chart(
        env=env,
        is_benchmark=False,
        episode_num=(episode_num + 1),
        cum_reward=cum_reward,
        auto_open_html=True,
    )
    export_dbs(
        env=env,
        is_benchmark=False,
        episode_num=(episode_num + 1),
    )


def eval_agent_policy(
    # num_episodes: int,
    seeds: Sequence[int],
    sim_randomise_reset: bool,
) -> TestResults:
    env = make_env(
        ROOT_EXP_TYPE,
        tensorboard_path=None,
        normalise_obs=False,
        normalise_rewards=False,
        gantt_chart=True,
        seeds=seeds,
        verify_env=False,
        sim_randomise_reset=sim_randomise_reset,
    )

    if TEST_NORMALISE_OBS:
        pth_vec_norm = get_path_vec_norm()
        if not pth_vec_norm.exists():
            raise FileNotFoundError(f'VecNormalize info not found under: {pth_vec_norm}')
        env = VecNormalize.load(str(pth_vec_norm), env)
        env.norm_reward = False
        env.training = False
        print('Normalization info loaded successfully.')

    # model.set_env(env)
    model = load_model(env=env)

    loggers.base.info('[MODEL EVAL] Start evaluation...')
    num_episodes = len(seeds)
    episodes_cum_rewards, _episode_lengths = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=num_episodes,
        deterministic=True,
        use_masking=True,
        callback=callback,
        return_episode_rewards=True,
    )
    episodes_cum_rewards = cast(list[float], episodes_cum_rewards)
    seeds_used = cast(list[int | None], env.get_attr('seeds_used')[0])
    # if terminated VecEnvs are reset automatically (do not count these)
    seeds_used = seeds_used[:-1]
    mean_episode_reward = cast(float, np.mean(episodes_cum_rewards))
    std_episode_reward = cast(float, np.std(episodes_cum_rewards))
    results: TestResults = TestResults(
        seeds=seeds_used,
        rewards=episodes_cum_rewards,
        rewards_mean=mean_episode_reward,
        rewards_std=std_episode_reward,
    )
    print(f'[AGENT EVAL] Rewards for {num_episodes} episodes: \n\t{episodes_cum_rewards}')
    print(
        f'[AGENT EVAL] Mean reward: {mean_episode_reward:.4f} - Num episodes: {num_episodes}'
    )

    return results


def eval_agent_benchmark(
    # num_episodes: int,
    seeds: Sequence[int],
    sim_randomise_reset: bool,
) -> TestResults:
    env = JSSEnv(
        VAL_EXP_TYPE,
        agent_type=DEC_TYPE,
        seeds=seeds,
        sim_randomise_reset=sim_randomise_reset,
        gantt_chart_on_termination=True,
        sim_check_agent_feasibility=True,
        builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
    )
    loggers.base.info('[MODEL EVAL - ValAgent] Start evaluation with validation agent...')
    episodes_cum_rewards: list[float] = []
    seed = get_relevant_seed(ROOT_RNG_SEEDS)
    _obs, _ = env.reset(seed=seed)

    num_episodes = len(seeds)
    for episode_num in range(num_episodes):
        if episode_num != 0:
            _obs, _ = env.reset()
        episode_rewards: list[float] = []
        episode_actions: list[int] = []
        terminated: bool = False
        truncated: bool = False
        val_agent = cast('ValidateAllocationAgent | ValidateSequencingAgent', env.agent)

        while not (terminated or truncated):
            action = val_agent.simulate_decision_making()
            _obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            episode_actions.append(action)

        cum_reward = sum(episode_rewards)
        episodes_cum_rewards.append(cum_reward)
        print(f'[BENCHMARK] Episode {episode_num + 1}: Reward = {cum_reward:.4f}')

        export_gantt_chart(
            env=env,
            is_benchmark=True,
            episode_num=(episode_num + 1),
            cum_reward=cum_reward,
            auto_open_html=True,
        )
        export_dbs(
            env=env,
            is_benchmark=True,
            episode_num=(episode_num + 1),
        )

    seeds_used = env.seeds_used
    mean_episode_reward = cast(float, np.mean(episodes_cum_rewards))
    std_episode_reward = cast(float, np.std(episodes_cum_rewards))
    results: TestResults = TestResults(
        seeds=seeds_used,
        rewards=episodes_cum_rewards,
        rewards_mean=mean_episode_reward,
        rewards_std=std_episode_reward,
    )
    print(f'[BENCHMARK] Rewards for {num_episodes} episodes: \n\t{episodes_cum_rewards}')
    print(
        f'[BENCHMARK]: Cycle Time = {env.cycle_time}, Utilisation = {env.sim_utilisation:.4%}'
    )
    print(
        f'[BENCHMARK] Mean reward: {mean_episode_reward:.4f} - Num episodes: {num_episodes}'
    )

    return results


def benchmark() -> None:
    env = JSSEnv(
        ROOT_EXP_TYPE,
        agent_type=DEC_TYPE,
        seeds=ROOT_RNG_SEEDS,
        sim_randomise_reset=False,
        sim_check_agent_feasibility=True,
        builder_func_family=BuilderFuncFamilies.SINGLE_PRODUCTION_AREA,
    )

    sim_env = env.run_without_agent()
    cycle_time = sim_env.dispatcher.cycle_time
    title_benchmark = (
        f'Gantt Chart<br>Benchmark<br>ExpType: {ROOT_EXP_TYPE}, cycle time: {cycle_time}'
    )
    _ = sim_env.dispatcher.draw_gantt_chart(
        save_html=True,
        title=title_benchmark,
        filename='Benchmark',
        base_folder=ROOT_FOLDER,
        sort_by_proc_station=True,
    )


def run_agent() -> TestResults:
    results_agent = eval_agent_policy(seeds=ROOT_RNG_SEEDS, sim_randomise_reset=False)
    export_results(results_agent, 'results_agent')

    return results_agent


def run_benchmark() -> TestResults:
    results_bench = eval_agent_benchmark(seeds=ROOT_RNG_SEEDS, sim_randomise_reset=False)
    export_results(results_bench, 'results_benchmark')

    return results_bench


def main() -> None:
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message=r'^[\s]*.*to get variables from other wrappers is deprecated.*$',
    )
    t1 = time.perf_counter()
    if TEST_PERFORM_AGENT:
        # eval_agent_policy(
        #     num_episodes=TEST_NUM_EPISODES, seeds=ROOT_RNG_SEEDS, sim_randomise_reset=False
        # )
        results_agent = run_agent()
    print('--------------------------------------------------------------------')
    if TEST_PERFORM_BENCHMARK:
        # eval_agent_benchmark(
        #     num_episodes=TEST_NUM_EPISODES, seeds=ROOT_RNG_SEEDS, sim_randomise_reset=False
        # )
        results_bench = run_benchmark()
    t2 = time.perf_counter()
    dur = t2 - t1
    print(f'Duration for run: {dur:.4f} sec')
    print('######## Results of run ######## ')
    print('Agent: ', results_agent)
    print('Benchmark: ', results_bench)


if __name__ == '__main__':
    main()
