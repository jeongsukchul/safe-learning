import functools
import jax
import jax.numpy as jnp
from brax.envs import Wrapper
from mujoco_playground import MjxEnv, State, dm_control_suite

from ss2r.benchmark_suites.rewards import tolerance
_POLE_ID = -1  # your pole body/geom index (as in your original)
# NOTE: In MuJoCo, geom indices and body indices are different.
# Your original code uses _POLE_ID for both sys.geom_size and sys.body_mass etc.
# This rewrite preserves that behavior exactly.

def domain_randomization(cfg, sys, params=None, rng: jax.Array = None):
    """
    Format:
      - dist(...) if rng is not None
      - shift_dynamics(params): apply given params (vmap)
      - rand_dynamics(rng): sample params then apply (vmap)
      - build in_axes + sys.tree_replace once
    Returns: (sys, in_axes, packed_params)
    where packed_params = [pole_length_sample, mass_sample, gear_sample]
    """

    # ---- distribution for sampling (only used in rng branch) ----
    if rng is not None:
        dr_low = jnp.array([cfg.pole_length[0], cfg.pole_mass[0], cfg.gear[0]])
        dr_high = jnp.array([cfg.pole_length[1], cfg.pole_mass[1], cfg.gear[1]])
        dist = functools.partial(
            jax.random.uniform,
            shape=(dr_low.shape[0],),
            minval=dr_low,
            maxval=dr_high,
        )

    # ---- common "apply params -> updated fields" logic ----
    def _apply_params(p):
        """
        p = [pole_length_sample, mass_sample, gear_sample]
        returns updated (body_ipos, body_mass, body_inertia, geom_size, actuator_gear)
        """
        pole_length_sample, mass_sample, gear_sample = p[0], p[1], p[2]

        # length scaling
        length = 0.5 + pole_length_sample
        scale_factor = length / 0.5

        # geom size update
        geom = sys.geom_size
        geom = geom.at[_POLE_ID, 1].set(length)

        # mass/inertia scale from length scaling
        mass = sys.body_mass
        mass = mass.at[_POLE_ID].multiply(scale_factor)

        inertia = sys.body_inertia
        inertia = inertia.at[_POLE_ID].multiply(scale_factor**3)

        # additional mass randomization (relative scale)
        scale = (sys.body_mass[_POLE_ID] + mass_sample) / sys.body_mass[_POLE_ID]
        mass = mass.at[_POLE_ID].multiply(scale)

        # NOTE: your original code does:
        #   inertia = sys.body_inertia.at[_POLE_ID].multiply(scale)
        # i.e., it re-reads sys.body_inertia (NOT the already length-scaled inertia).
        # That effectively discards the length scaling for inertia. Probably unintended,
        # but we preserve it exactly.
        inertia = sys.body_inertia.at[_POLE_ID].multiply(scale)

        # COM shift / inertia position shift
        inertia_pos = sys.body_ipos
        inertia_pos = inertia_pos.at[_POLE_ID, -1].add(pole_length_sample / 2.0)

        # actuator gear shift
        gear = sys.actuator_gear
        gear = gear.at[0, 0].add(gear_sample)

        return inertia_pos, mass, inertia, geom, gear

    # ---- vmap branches (format matching your example) ----
    def shift_dynamics(params_vec):
        # params_vec is shape (3,)
        inertia_pos, mass, inertia, geom, gear = _apply_params(params_vec)
        return inertia_pos, mass, inertia, geom, gear

    def rand_dynamics(rng_i):
        # sample params for this rng_i
        p = dist(rng_i)  # shape (3,)
        inertia_pos, mass, inertia, geom, gear = _apply_params(p)
        return inertia_pos, mass, inertia, geom, gear, p

    # ---- select branch ----
    if rng is None and params is not None:
        # params expected shape: (B, 3)
        inertia_pos, mass, inertia, geom, gear = shift_dynamics(params)
        packed = params
    elif rng is not None and params is None:
        # rng expected shape: (B, 2) PRNGKey or (B,) keys depending on your usage
        inertia_pos, mass, inertia, geom, gear, packed = rand_dynamics(rng)
    else:
        raise ValueError("Provide exactly one of (rng, params).")

    # ---- in_axes + sys.tree_replace (exact pattern) ----
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "geom_size": 0,
            "actuator_gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_mass": mass,
            "body_inertia": inertia,
            "body_ipos": inertia_pos,
            "geom_size": geom,
            "actuator_gear": gear,
        }
    )

    return sys, in_axes#, packed

class ConstraintWrapper(Wrapper):
    def __init__(self, env: MjxEnv, slider_position_bound: float):
        super().__init__(env)
        self.slider_position_bound = slider_position_bound

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        slider_pos = state.data.qpos[self.env._slider_qposadr]
        cost = (jnp.abs(slider_pos) >= self.slider_position_bound).astype(jnp.float32)
        nstate.info["cost"] = cost
        return nstate


class ActionCostWrapper(Wrapper):
    def __init__(self, env: MjxEnv, action_cost_scale: float):
        super().__init__(env)
        self.action_cost_scale = action_cost_scale

    def step(self, state: State, action: jax.Array) -> State:
        action_cost = (
            self.action_cost_scale * (1 - tolerance(action, (-0.1, 0.1), 0.1))[0]
        )
        nstate = self.env.step(state, action)
        nstate = nstate.replace(reward=nstate.reward - action_cost)
        return nstate


_envs = [
    env_name
    for env_name in dm_control_suite.ALL_ENVS
    if env_name.startswith("Cartpole")
]


def make_safe(name, **kwargs):
    limit = kwargs["config"]["slider_position_bound"]
    env = dm_control_suite.load(name, **kwargs)
    env = ConstraintWrapper(env, limit)
    return env


def make_hard(name, **kwargs):
    scale = kwargs["config"]["action_cost_scale"]
    env = dm_control_suite.load(name, **kwargs)
    env = ActionCostWrapper(env, scale)
    return env


for env_name in _envs:
    dm_control_suite.register_environment(
        f"Safe{env_name}",
        functools.partial(make_safe, env_name),
        dm_control_suite.cartpole.default_config,
    )

for env_name in _envs:
    dm_control_suite.register_environment(
        f"Hard{env_name}",
        functools.partial(make_hard, env_name),
        dm_control_suite.cartpole.default_config,
    )