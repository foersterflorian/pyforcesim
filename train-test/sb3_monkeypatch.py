import multiprocessing.connection as mp_con
from typing import Any, Dict, Optional

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def worker(
    remote: mp_con.Connection,
    parent_remote: mp_con.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info['TimeLimit.truncated'] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == 'reset':
                maybe_options = {'options': data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                # make sure retrieved attribute is not a callable (in this case a method)
                # since these are not pickable and crash the whole program
                attr = env.get_wrapper_attr(data)
                send_value = attr
                if callable(attr):
                    send_value = True
                remote.send(send_value)
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == 'is_wrapped':
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f'`{cmd}` is not implemented in the worker')
        except EOFError:
            break
