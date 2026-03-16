from functools import partial

import jax
import jax.numpy as jnp
from brax.envs import Env, Wrapper
from brax.envs.base import State
from mujoco_playground import dm_control_suite

_TORSO_ID = 1


def domain_randomization(cfg, sys, params=None, rng: jax.Array = None):
    # @jax.vmap
    # def randomize(rng):
    #     #  https://github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/environments/walker.py#L593
    #     torso_length_sample = jax.random.uniform(
    #         rng, minval=cfg.torso_length[0], maxval=cfg.torso_length[1]
    #     )
    #     length = 0.3 + torso_length_sample
    #     scale_factor = length / 0.3
    #     # Make scale factor closer to 1 (either from above or below)
    #     # to help simulation stability.
    #     scale_factor = 3 * scale_factor / (2 * scale_factor + 1)
    #     geom = sys.geom_size.copy()
    #     # geom = geom.at[_TORSO_ID, 1].set(length)
    #     inertia_pos = sys.body_ipos.copy()
    #     inertia_pos = inertia_pos.at[_TORSO_ID, -1].add(torso_length_sample / 2.0)
    #     # mass = sys.body_mass.at[_TORSO_ID].multiply(scale_factor)
    #     # inertia = sys.body_inertia.at[_TORSO_ID].multiply(scale_factor**3)
    #     mass = sys.body_mass
    #     inertia = sys.body_inertia
    #     friction_sample = jax.random.uniform(
    #         rng, minval=cfg.friction[0], maxval=cfg.friction[1]
    #     )
    #     friction = sys.geom_friction.at[:, 0].add(friction_sample)
    #     damping_sample = jax.random.uniform(
    #         rng, minval=cfg.joint_damping[0], maxval=cfg.joint_damping[1]
    #     )
    #     damping = sys.dof_damping.at[3:].add(damping_sample)
    #     gear = sys.actuator_gear.copy()
    #     gear_sample = jax.random.uniform(rng, minval=cfg.gear[0], maxval=cfg.gear[1])
    #     gear = gear.at[:, 0].add(gear_sample)
    #     return (
    #         inertia_pos,
    #         mass,
    #         inertia,
    #         geom,
    #         friction,
    #         damping,
    #         gear,
    #         jnp.hstack(
    #             [friction_sample, torso_length_sample, damping_sample, gear_sample]
    #         ),
    #     )

    # ---- distribution for sampling (only used in rng branch) ----
    if rng is not None:
        dr_low = jnp.array([cfg.torso_length[0], cfg.friction[0], cfg.joint_damping[0], cfg.gear[0]])
        dr_high = jnp.array([cfg.torso_length[1], cfg.friction[1], cfg.joint_damping[1], cfg.gear[1]])
        dist = partial(
            jax.random.uniform,
            shape=(dr_low.shape[0],),
            minval=dr_low,
            maxval=dr_high,
        )

    def _apply_params(p):
        """p = [torso_length, friction, joint_damping, gear] — same as AdV uniform samples."""
        
        torso_length_sample = p[0]
        inertia_pos = sys.body_ipos.copy()
        inertia_pos = inertia_pos.at[_TORSO_ID, -1].add(torso_length_sample / 2.0)

        mass = sys.body_mass
        inertia = sys.body_inertia
        geom = sys.geom_size.copy()


        friction_sample = p[1]
        friction = sys.geom_friction.at[:, 0].add(friction_sample)

        damping_sample = p[2]
        damping = sys.dof_damping.at[3:].add(damping_sample)
        gear = sys.actuator_gear.copy()
        gear_sample = p[3]
        gear = gear.at[:, 0].add(gear_sample)
        return inertia_pos, mass, inertia, geom, friction, damping, gear

    def shift_dynamics(params_vec):
        # params_vec is shape (3,)
        inertia_pos, mass, inertia, geom, friction, damping, gear= _apply_params(params_vec)
        return inertia_pos, mass, inertia, geom, friction, damping, gear

    def rand_dynamics(rng_i):  # sample params for this rng_i
        p = dist(rng_i)  #  샘플링할 distribution
        inertia_pos, mass, inertia, geom, friction, damping, gear = _apply_params(p)
        return inertia_pos, mass, inertia, geom, friction, damping, gear, p


    # ---- select branch ----
    if rng is None and params is not None: # Param
        inertia_pos, mass, inertia, geom, friction, damping, gear = shift_dynamics(params) # samples는 뭐지?
        packed = params
    elif rng is not None and params is None: # RNG, random token
        inertia_pos, mass, inertia, geom, friction, damping, gear, packed = rand_dynamics(rng)
    else:
        raise ValueError("Provide exactly one of (rng, params).")

    # inertia_pos, mass, inertia, geom, friction, damping, gear, samples = randomize(rng) # samples는 뭔지?
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_ipos": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "geom_size": 0,
            "geom_friction": 0,
            "dof_damping": 0,
            "actuator_gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_ipos": inertia_pos,
            "body_mass": mass,
            "body_inertia": inertia,
            "geom_size": geom,
            "geom_friction": friction,
            "dof_damping": damping,
            "actuator_gear": gear,
        }
    )
    return sys, in_axes# , samples


class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env, limit: float):
        assert isinstance(env, dm_control_suite.walker.PlanarWalker)
        super().__init__(env)
        self.limit = limit

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        joint_velocities = nstate.data.qvel[3:]
        cost = jnp.less(jnp.max(jnp.abs(joint_velocities)), self.limit).astype(
            jnp.float32
        )
        nstate.info["cost"] = cost
        return nstate


for run in [True, False]:

    def make(run, **kwargs):
        run_str = "Run" if run else "Walk"
        angular_velocity_limit = kwargs.pop("joint_velocity_limit", 16.25)
        env = dm_control_suite.load(f"Walker{run_str}", **kwargs)
        env = ConstraintWrapper(env, angular_velocity_limit)
        return env

    run_str = "Run" if run else "Walk"
    name_str = f"SafeWalker{run_str}"
    dm_control_suite.register_environment(
        name_str, partial(make, run=run), dm_control_suite.walker.default_config
    )
