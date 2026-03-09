import time
from typing import Any, Callable, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from brax.envs.wrappers.training import EvalMetrics, EvalWrapper
from brax.training.acting import Evaluator, generate_unroll
from brax.training.types import Metrics, Policy, PolicyParams, PRNGKey, Transition



class ConstraintEvalWrapper(EvalWrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        reset_state.metrics["cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(state, action)
        if "eval_reward" in nstate.info:
            reward = nstate.info["eval_reward"]
        else:
            reward = nstate.reward
        nstate.metrics["reward"] = reward
        nstate.metrics["cost"] = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info.get("steps", jnp.zeros_like(state_metrics.episode_steps)),
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)
        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info["eval_metrics"] = eval_metrics
        return nstate
class ConstraintAdvEvalWrapper(EvalWrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        reset_state.metrics["cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array, params) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(state, action, params)
        if "eval_reward" in nstate.info:
            reward = nstate.info["eval_reward"]
        else:
            reward = nstate.reward
        nstate.metrics["reward"] = reward
        print("n state info cost", nstate.info['cost'])
        nstate.metrics["cost"] = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info.get("steps", jnp.zeros_like(state_metrics.episode_steps)),
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)
        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info["eval_metrics"] = eval_metrics
        return nstate

class InterventionConstraintEvalWrapper(EvalWrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        reset_state.metrics["cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        reset_state.metrics["intervention"] = jnp.zeros_like(reset_state.reward)
        episode_metrics = jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics)
        episode_metrics["max_policy_distance"] = jnp.zeros_like(reset_state.reward)
        episode_metrics["max_safety_gap"] = jnp.zeros_like(reset_state.reward)
        episode_metrics["max_expected_total_cost"] = jnp.zeros_like(reset_state.reward)
        episode_metrics["max_cumulative_cost"] = jnp.zeros_like(reset_state.reward)
        episode_metrics["max_q_c"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["intervention"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["policy_distance"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["safety_gap"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["expected_total_cost"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["cumulative_cost"] = jnp.zeros_like(reset_state.reward)
        reset_state.info["q_c"] = jnp.zeros_like(reset_state.reward)
        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        nstate.metrics["cost"] = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        nstate.metrics["intervention"] = nstate.info.get(
            "intervention", jnp.zeros_like(nstate.reward)
        )
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info.get("steps", jnp.zeros_like(state_metrics.episode_steps)),
            state_metrics.episode_steps,
        )
        episode_metrics = {}
        for k, v in state_metrics.episode_metrics.items():
            if k in [
                "max_policy_distance",
                "max_safety_gap",
                "max_expected_total_cost",
                "max_cumulative_cost",
                "max_q_c",
            ]:
                episode_metrics[k] = jnp.maximum(
                    nstate.info.get(k.strip("max_"), jnp.zeros_like(nstate.reward))
                    * state_metrics.active_episodes,
                    v,
                )
            else:
                episode_metrics[k] = (
                    v + nstate.metrics[k] * state_metrics.active_episodes
                )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)
        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info["eval_metrics"] = eval_metrics
        return nstate


class ConstraintsEvaluator(Evaluator):
    def __init__(
        self,
        eval_env: Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],  # type: ignore
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: jax.Array,
        budget: float,
        num_episodes: int = 10,
    ):
        self._key = key
        self._eval_walltime = 0.0
        eval_env = ConstraintEvalWrapper(eval_env)
        self.budget = budget
        self.num_episodes = num_episodes

        def generate_eval_unroll(policy_params: PolicyParams, key: PRNGKey) -> State:  # type: ignore
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            
            return generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
            )[0]

        self._generate_eval_unroll = jax.jit(
            jax.vmap(generate_eval_unroll, in_axes=(None, 0))
        )
        self._steps_per_unroll = episode_length * num_eval_envs * num_episodes

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
        prefix: str = "eval",
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)
        unroll_key = jax.random.split(unroll_key, self.num_episodes)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        constraint = eval_state.info["eval_metrics"].episode_metrics["cost"].mean(0)
        eval_state.info["eval_metrics"].episode_metrics["cost"] = constraint
        safe = np.where(constraint < self.budget, 1.0, 0.0)
        eval_state.info["eval_metrics"].episode_metrics["safe"] = safe
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"{prefix}/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value  # type: ignore
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics[f"{prefix}/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics[f"{prefix}/epoch_eval_time"] = epoch_eval_time
        metrics[f"{prefix}/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            f"{prefix}/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }
        return metrics
class TransitionwithParams(NamedTuple):
    """Transition with additional dynamics parameters."""
    observation: jax.Array
    dynamics_params: jax.Array
    action: jax.Array
    reward: jax.Array
    discount: jax.Array
    next_observation: jax.Array
    extras: Dict[str, Any] = {}


def adv_step(
    env: Env,
    env_state: State,
    dynamics_params: jnp.ndarray,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
    ):
    
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions, dynamics_params)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        # dynamics_params=dynamics_params,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        )
def generate_adv_unroll(
    env: Env,
    env_state: State,
    dynamics_params :jnp.ndarray,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    non_stationary: bool=False,
):
    """Collect trajectories of given unroll_length."""
    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        if non_stationary:
            nstate, transition = adv_step(
                env, state, unused_t, policy, current_key, extra_fields=extra_fields
            )
        else:
            nstate, transition = adv_step(
                env, state, dynamics_params, policy, current_key, extra_fields=extra_fields
            )
        return (nstate, next_key), transition

    if non_stationary:
        (final_state, _), data = jax.lax.scan(
            f, (env_state, key), (dynamics_params), length=unroll_length
        )
    else:
        (final_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
    return final_state, data

class ConstraintsAdvEvaluator(Evaluator):
    def __init__(
        self,
        eval_env: Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],  # type: ignore
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: jax.Array,
        budget: float,
        num_episodes: int = 10,
        non_stationary = False,
    ):
        self._key = key
        self._eval_walltime = 0.0
        eval_env = ConstraintAdvEvalWrapper(eval_env)
        self.budget = budget
        self.num_episodes = num_episodes

        def generate_eval_unroll(policy_params: PolicyParams, eval_params, key: PRNGKey) -> State:  # type: ignore
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_adv_unroll(
                eval_env,
                eval_first_state,
                eval_params,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
                non_stationary=non_stationary,
            )[0]
            # return generate_unroll(
            #     eval_env,
            #     eval_first_state,
            #     eval_policy_fn(policy_params),
            #     key,
            #     unroll_length=episode_length // action_repeat,
            # )[0]
        
        # self._generate_eval_unroll = jax.jit(
        #     jax.vmap(generate_eval_unroll, in_axes=(None, None, 0))
        # )
        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs * num_episodes

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        dynamics_params : jnp.ndarray,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
        prefix: str = "eval",
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)
        unroll_key = jax.random.split(unroll_key, self.num_episodes)

        t = time.time()
        def f(carry, key):
            eval_state = self._generate_eval_unroll(policy_params, dynamics_params, key)
            # eval_metrics = eval_state.info['eval_metrics']
            return carry, eval_state
        _, eval_state = jax.lax.scan(f, None, unroll_key)
        # eval_state = self._generate_eval_unroll(policy_params, dynamics_params, unroll_key)
        constraint = eval_state.info["eval_metrics"].episode_metrics["cost"].mean(0)
        eval_state.info["eval_metrics"].episode_metrics["cost"] = constraint
        safe = np.where(constraint < self.budget, 1.0, 0.0)
        eval_state.info["eval_metrics"].episode_metrics["safe"] = safe
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"{prefix}/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value  # type: ignore
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics[f"{prefix}/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics[f"{prefix}/epoch_eval_time"] = epoch_eval_time
        metrics[f"{prefix}/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            f"{prefix}/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }
        return metrics

