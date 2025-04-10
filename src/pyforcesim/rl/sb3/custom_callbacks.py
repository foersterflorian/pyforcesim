import os

import numpy as np
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from pyforcesim.loggers import gym_env as logger


class MaskableEvalCallback(EvalCallback):
    """
    copied from StableBaselines3 Contrib package, enhanced with customised behaviour

    Callback for evaluating an agent. Supports invalid action masking.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param use_masking: Whether to use invalid action masks during evaluation
    """

    def __init__(self, *args, use_masking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_masking = use_masking
        self.continue_training = True

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        'Training and eval env are not wrapped the same way, '
                        'see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback '
                        'and warning above.'
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self.eval_env.training = False  # type: ignore

            # Note that evaluate_policy() has been patched to support masking
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            logger.info(
                'Eval all rewards: %s',
                episode_rewards,
            )
            logger.info(
                'Eval num_timesteps=%d, episode_reward=%.2f +/- %.2f',
                self.num_timesteps,
                mean_reward,
                std_reward,
            )
            logger.info('Episode length: %.2f +/- %.2f', mean_ep_length, std_ep_length)
            # Add to current Logger
            self.logger.record('eval/mean_reward', float(mean_reward))
            self.logger.record('eval/mean_ep_length', mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f'Success rate: {100 * success_rate:.2f}%')
                self.logger.record('eval/success_rate', success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                'time/total_timesteps', self.num_timesteps, exclude='tensorboard'
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print('New best mean reward!')
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                    vec_norm = self.model.get_vec_normalize_env()
                    if vec_norm is not None:
                        vec_norm.save(
                            os.path.join(self.best_model_save_path, 'best_model_vec_norm.pkl')
                        )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
                    self.continue_training = continue_training

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
