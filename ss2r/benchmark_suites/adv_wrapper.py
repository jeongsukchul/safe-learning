import contextlib
import functools
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from mujoco_playground._src import mjx_env
from ss2r.benchmark_suites import wrappers
from mujoco_playground import wrapper as mujoco_playground_wrapper


from brax.envs.wrappers import training as brax_training


def wrap_for_adv_training(
    env: mjx_env.MjxEnv,
    param_size: int | None = None,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
    dr_range_low: jnp.ndarray = None,
    dr_range_high: jnp.ndarray = None,
    get_grad: bool = False,
    augment_state: bool = False,
    hard_resets: bool = False,
) -> Wrapper:
    """Common wrapper pattern for all brax training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized model
        and in_axes to vmap over
    """
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
        env = NoRandomizationVmapWrapper(env)
    else:
        if param_size is None or param_size <= 0:
            raise ValueError("param_size must be positive when randomization is enabled")
        env = AdVmapWrapper(
            env,
            randomization_fn,
            param_size,
            dr_range_low,
            dr_range_high,
            get_grad,
            augment_state,
        )
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    if hard_resets:
        env = HardAutoResetWrapper(env)
    else:
        env = BraxAutoResetWrapper(env)
    return env


class NoRandomizationVmapWrapper(Wrapper):
    """Adapter to keep (state, action, params) signature without DR."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> State:
        del params
        return self.env.step(state, action)


class AdVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""
    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[System], Tuple[System, System]],
        param_size: int,
        dr_range_low: jnp.ndarray = None,
        dr_range_high: jnp.ndarray = None,
        get_grad: bool = False,
        augment_state: bool = False,
    ):
        super().__init__(env)
        # randomization_fn is expected to be called like:
        #   randomization_fn(model=..., rng=None, params=...)
        self.rand_fn = functools.partial(randomization_fn, sys=self.mjx_model, rng=None)

        self.get_grad = get_grad
        self.param_size = param_size
        self.dr_range_low = dr_range_low
        self.dr_range_high = dr_range_high
        self.augment_state = augment_state

    @contextlib.contextmanager
    def v_env_fn(self, mjx_model: mjx.Model):
        base = self.env.unwrapped
        old_model = base._mjx_model
        try:
            base._mjx_model = mjx_model
            yield self.env
        finally:
            base._mjx_model = old_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def dr_reset(rng_i):
            param_rng, env_rng = jax.random.split(rng_i)
            params = jax.random.uniform(
                param_rng,
                (self.param_size,),
                minval=self.dr_range_low,
                maxval=self.dr_range_high,
            )
            mjx_model, inaxes = self.rand_fn(params=params)
            with self.v_env_fn(mjx_model) as v_env:
                base = self.env.unwrapped
                assert base._mjx_model is mjx_model
                return v_env.reset(env_rng), params

        state, params = jax.vmap(dr_reset)(rng)
        state.info["dr_params"] = params
        if self.augment_state:
            state = self._add_privileged_state(state)
        if self.get_grad:
            # Note: state.obs might be a pytree; your original used tree_map on state.obs.
            state.info["grad"] = jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape + (self.param_size,), dtype=x.dtype),
                state.obs,
            )
        return state
    def _add_privileged_state(self, state):
        """Adds privileged state to the observation if augmentation is enabled."""
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs={
                    "state": state.obs,
                    "privileged_state": jnp.concatenate(
                        [state.obs, state.info['dr_params']], -1
                    ),
                }
            )
        else:
            state = state.replace(
                obs={
                    "state": state.obs["state"],
                    "privileged_state": jnp.concatenate(
                        [state.obs["privileged_state"], state.info['dr_params']], -1
                    ),
                }
            )
        return state
    def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> State:
        def _step(params_i, s_i, a_i):
            if params_i is None:
                return self.env.step(s_i, a_i)
            mjx_model, inaxes = self.rand_fn(params=params_i)
            with self.v_env_fn(mjx_model) as v_env:
                base = self.env.unwrapped
                assert base._mjx_model is mjx_model
                return v_env.step(s_i, a_i)

        ns = jax.vmap(_step)(params, state, action)
        ns.info["dr_params"] = params
        return ns
    @property
    def observation_size(self):
        """Compute observation size based on the augmentation setting."""
        if not self.augment_state:
            return self.env.observation_size

        if isinstance(self.env.observation_size, int):
            return {
                "state": (self.env.observation_size,),
                "privileged_state": (
                    self.env.observation_size + self.domain_parameters.shape[1],
                ),
            }
        else:
            return {
                "state": (self.env.observation_size["state"],),
                "privileged_state": (
                    self.env.observation_size["privileged_state"]
                    + self.domain_parameters.shape[1],
                ),
            }


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1])

        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info["episode_done"] = jnp.zeros(rng.shape[:-1])

        episode_metrics = {
            "sum_reward": jnp.zeros(rng.shape[:-1]),
            "length": jnp.zeros(rng.shape[:-1]),
        }
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])

        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array, params: jax.Array) -> State:
        def f(carry_state, _):
            nstate = self.env.step(carry_state, action, params)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        rewards = jnp.sum(rewards, axis=0)
        state = state.replace(reward=rewards)

        steps = state.info["steps"] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)

        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jnp.where(steps >= episode_length, 1 - state.done, zero)
        state.info["steps"] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] += rewards
        state.info["episode_metrics"]["sum_reward"] *= (1 - prev_done)

        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["length"] *= (1 - prev_done)

        for metric_name in state.metrics.keys():
            if metric_name != "reward":
                state.info["episode_metrics"][metric_name] += state.metrics[metric_name]
                state.info["episode_metrics"][metric_name] *= (1 - prev_done)

        state.info["episode_done"] = done
        return state.replace(done=done)


class CostEpisodeWrapper(EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""

    def step(self, state: State, action: jax.Array, params) -> State:
        def f(state, _):
            nstate = self.env.step(state, action, params)
            maybe_cost = nstate.info.get("cost", None)
            maybe_eval_reward = nstate.info.get("eval_reward", None)
            return nstate, (nstate.reward, maybe_cost, maybe_eval_reward)

        state, (rewards, maybe_costs, maybe_eval_rewards) = jax.lax.scan(
            f, state, (), self.action_repeat
        )
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        if maybe_costs is not None:
            state.info["cost"] = jnp.sum(maybe_costs, axis=0)
        if maybe_eval_rewards is not None:
            state.info["eval_reward"] = jnp.sum(maybe_eval_rewards, axis=0)
        steps = state.info["steps"] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)
class HardAutoResetWrapper(Wrapper):
    """Automatically reset Brax envs that are done.

    Resample only when >=1 environment is actually done. Still resamples for all
    """

    def reset(self, rng: jax.Array) -> State :
        rng, sample_rng = jax.vmap(jax.random.split, out_axes=1)(rng)
        state = self.env.reset(sample_rng)
        state.info["reset_rng"] = rng
        return state

    def step(self, state: State, action: jax.Array, params) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action, params)

        if hasattr(state, "pipeline_state"):
            data_name = "pipeline_state"
        elif hasattr(state, "data"):
            data_name = "data"
        else:
            raise NotImplementedError

        def safe_reset(rng):
            nstate = self.reset(rng)
            return nstate.obs, getattr(state, data_name)

        obs, new_data = jax.lax.cond(
            state.done.any(),
            safe_reset,
            lambda rng: (state.obs, getattr(state, data_name)),
            state.info["reset_rng"],
        )
        return state.replace(**{data_name: new_data, "obs": obs})

class BraxAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def __init__(self, env: Any, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = "AutoResetWrapper"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)

        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = rng
        state.info[f"{self._info_key}_done_count"] = jnp.zeros(key.shape[:-1], dtype=int)
        return state

    def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> mjx_env.State:
        reset_state = None

        rng_key = jax.vmap(jax.random.split)(state.info[f"{self._info_key}_rng"])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f"{self._info_key}_first_data"]
            reset_obs = state.info[f"{self._info_key}_first_obs"]

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        # clear done before stepping (matches your original)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action, params)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f"{self._info_key}_done_count"

        if self._full_reset and reset_state is not None:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]

            if "steps" in next_info:
                next_info["steps"] = state.info["steps"]

            preserve_info_key = f"{self._info_key}_preserve_info"
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f"{self._info_key}_rng"] = reset_rng

        return state.replace(data=data, obs=obs, info=next_info)