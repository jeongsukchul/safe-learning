import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey, Transition

from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.evaluation import adv_step
from ss2r.rl.types import MakePolicyFn, UnrollFn
from ss2r.rl.utils import quantize_images, remove_pixels


def get_collection_fn(cfg):
    if cfg.agent.data_collection.name == "step":
        return collect_adv_single_step
    elif cfg.agent.data_collection.name == "episodic":

        def generate_episodic_unroll(
            env,
            env_state,
            make_policy_fn,
            policy_params,
            key,
            extra_fields,
        ):
            env_state, transitions = acting.generate_unroll(
                env,
                env_state,
                make_policy_fn(policy_params),
                key,
                cfg.training.episode_length,
                extra_fields,
            )
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
            return env_state, transitions

        return make_collection_fn(generate_episodic_unroll)
    else:
        raise ValueError(f"Unknown data collection {cfg.agent.data_collection.name}")


def actor_step(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.actor_step(env, env_state, policy, key, extra_fields)



def generate_unroll(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    unroll_length,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.generate_unroll(
        env, env_state, policy, key, unroll_length, extra_fields
    )


def make_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn,
        params: Params,
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env,
            env_state,
            make_policy_fn,
            (normalizer_params, params),
            key,
            extra_fields=extra_fields,
        )
        normalizer_params = running_statistics.update(
            normalizer_params, remove_pixels(transitions.observation)
        )
        transitions = float16(transitions)
        transitions = transitions._replace(
            observation=quantize_images(transitions.observation),
            next_observation=quantize_images(transitions.next_observation),
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(actor_step)
def make_adv_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn,
        params: Params,
        normalizer_params: running_statistics.RunningStatisticsState,
        dynamics_params,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env,
            env_state,
            dynamics_params,
            make_policy_fn,
            (normalizer_params, params),
            key,
            extra_fields=extra_fields,
        )
        normalizer_params = running_statistics.update(
            normalizer_params, remove_pixels(transitions.observation)
        )
        transitions = float16(transitions)
        transitions = transitions._replace(
            observation=quantize_images(transitions.observation),
            next_observation=quantize_images(transitions.next_observation),
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    return collect_data
def actor_adv_step(
    env,
    env_state,
    dynamics_params,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return adv_step(env, env_state, dynamics_params, policy, key, extra_fields)

collect_adv_single_step = make_adv_collection_fn(actor_adv_step)