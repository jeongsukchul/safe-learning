from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, State
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.safety_gym.lidar as lidar

_XML_PATH = Path(__file__).parent / "assets" / "xmls" / "scene_mjx_flat_terrain.xml"

_FLOOR_GEOM_ID = 0
_CHASSIS_BODY_ID = 2
_WHEEL_GEOM_IDS = jnp.array([8, 9, 10, 11])
_WHEEL_DOF_SLICE = slice(6, 10)
_NAV_EXTENTS = (-3.5, -3.5, 3.5, 3.5)
_HAZARD_BOX_HALF_HEIGHT = 2e-2
_CAPSULE_ROTATE_TO_X = jnp.array([0.70710678, 0.0, 0.70710678, 0.0])
_CAPSULE_ROTATE_TO_Y = jnp.array([0.70710678, 0.70710678, 0.0, 0.0])


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.01,
        episode_length=1000,
        action_scale=1.0,
        max_torque=1.0,
        command_config=config_dict.create(
            a=[1.25, 1.25],
            b=[0.85, 0.45],
        ),
        reward_config=config_dict.create(
            tracking_sigma=0.25,
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                lin_vel_z=-0.05,
                ang_vel_xy=-0.05,
                tilt=-0.5,
                action_rate=-0.01,
                energy=-0.0005,
                termination=-1.0,
            ),
        ),
    )


def go_to_goal_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_scale=1.0,
        max_torque=1.0,
        seed=0,
        goal_size=0.35,
        goal_reward=1.0,
        num_hazards=12,
        num_obstacles=10,
        hazard_size=0.35,
        obstacle_size=0.18,
        robot_keepout=0.45,
        goal_keepout=0.6,
        hazard_keepout=0.55,
        obstacle_keepout=0.55,
        visualize_lidar=False,
    )


def domain_randomization(sys, cfg, params=None, rng: jax.Array = None):
    if rng is not None:
        dr_low = jnp.array(
            [
                cfg.floor_friction[0],
                cfg.wheel_friction[0],
                cfg.chassis_mass_delta[0],
                cfg.wheel_damping_scale[0],
            ]
        )
        dr_high = jnp.array(
            [
                cfg.floor_friction[1],
                cfg.wheel_friction[1],
                cfg.chassis_mass_delta[1],
                cfg.wheel_damping_scale[1],
            ]
        )

        def sample(one_rng):
            return jax.random.uniform(
                one_rng, shape=(4,), minval=dr_low, maxval=dr_high
            )
    elif params is None:
        raise ValueError("Provide exactly one of rng or params.")

    def _apply(p):
        floor_friction, wheel_friction, chassis_mass_delta, wheel_damping_scale = p

        geom_friction = sys.geom_friction
        geom_friction = geom_friction.at[_FLOOR_GEOM_ID, 0].set(floor_friction)
        geom_friction = geom_friction.at[_WHEEL_GEOM_IDS, 0].set(wheel_friction)

        body_mass = sys.body_mass.at[_CHASSIS_BODY_ID].add(chassis_mass_delta)

        dof_damping = sys.dof_damping
        dof_damping = dof_damping.at[_WHEEL_DOF_SLICE].set(
            sys.dof_damping[_WHEEL_DOF_SLICE] * wheel_damping_scale
        )
        return geom_friction, body_mass, dof_damping

    if params is not None:
        geom_friction, body_mass, dof_damping = _apply(params)
        packed = params
    else:
        packed = sample(rng)
        geom_friction, body_mass, dof_damping = _apply(packed)

    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_mass": 0,
            "dof_damping": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "dof_damping": dof_damping,
        }
    )
    return sys, in_axes


class JackalJoystick(MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self):
        self._base_body_id = self._mj_model.body("base_link").id
        self._cmd_a = jnp.array(self._config.command_config.a)
        self._cmd_b = jnp.array(self._config.command_config.b)
        self._init_q = jnp.array(self._mj_model.qpos0)

    def _sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        rng, key_time, key_cmd, key_keep = jax.random.split(rng, 4)
        time_until_next_cmd = jax.random.exponential(key_time) * 5.0
        steps_until_next_cmd = jnp.round(time_until_next_cmd / self.dt).astype(jnp.int32)
        cmd = jax.random.uniform(key_cmd, shape=(2,), minval=-self._cmd_a, maxval=self._cmd_a)
        keep_mask = (
            jax.random.bernoulli(key_keep, p=self._cmd_b).astype(cmd.dtype) * 2.0 - 1.0
        )
        return rng, cmd * keep_mask, steps_until_next_cmd

    def reset(self, rng: jax.Array) -> State:
        qpos = self._init_q
        qvel = jnp.zeros(self.mjx_model.nv)

        rng, key_xy, key_yaw, key_qvel = jax.random.split(rng, 4)
        qpos = qpos.at[:2].set(jax.random.uniform(key_xy, (2,), minval=-0.25, maxval=0.25))
        yaw = jax.random.uniform(key_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        cy = jnp.cos(yaw * 0.5)
        sy = jnp.sin(yaw * 0.5)
        qpos = qpos.at[3:7].set(jnp.array([cy, 0.0, 0.0, sy]))
        qvel = qvel.at[:6].set(jax.random.uniform(key_qvel, (6,), minval=-0.1, maxval=0.1))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        rng, command, steps_until_next_cmd = self._sample_command(rng)

        info = {
            "rng": rng,
            "command": command,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jnp.zeros(self.action_size),
            "cost": jnp.zeros(()),
        }
        metrics = {
            "reward/tracking_lin_vel": jnp.zeros(()),
            "reward/tracking_ang_vel": jnp.zeros(()),
            "reward/tilt": jnp.zeros(()),
        }
        obs = self._get_obs(data, info)
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = self._action_to_ctrl(action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        rng = state.info["rng"]
        steps_until_next_cmd = state.info["steps_until_next_cmd"] - 1

        def _resample(_):
            return self._sample_command(rng)

        def _keep(_):
            return rng, state.info["command"], steps_until_next_cmd

        rng, command, steps_until_next_cmd = jax.lax.cond(
            steps_until_next_cmd <= 0, _resample, _keep, operand=None
        )
        info = dict(state.info)
        info.update(
            rng=rng,
            command=command,
            steps_until_next_cmd=steps_until_next_cmd,
            last_act=action,
            cost=jnp.zeros(()),
        )

        reward = self._get_reward(data, action, state.info["last_act"], command, state.metrics)
        done = self._get_done(data)
        reward = reward + self._config.reward_config.scales.termination * done
        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, state.metrics, info)

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        left = action[0] * self._config.max_torque * self._config.action_scale
        right = action[1] * self._config.max_torque * self._config.action_scale
        return jnp.array([left, right, left, right])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        torso_quat = mjx_env.get_sensor_data(self.mj_model, data, "base_link_quat")
        world_lin_vel = mjx_env.get_sensor_data(
            self.mj_model, data, "base_link_subtreelinvel"
        )
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        body_lin_vel = rot.T @ world_lin_vel
        wheel_vel = data.qvel[_WHEEL_DOF_SLICE]
        gyro = mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro")
        return jnp.concatenate(
            [
                torso_quat,
                body_lin_vel,
                gyro,
                wheel_vel,
                info["command"],
                info["last_act"],
            ]
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        last_action: jax.Array,
        command: jax.Array,
        metrics: dict[str, Any],
    ) -> jax.Array:
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        world_lin_vel = mjx_env.get_sensor_data(
            self.mj_model, data, "base_link_subtreelinvel"
        )
        body_lin_vel = rot.T @ world_lin_vel
        ang_vel = data.qvel[3:6]

        sigma = self._config.reward_config.tracking_sigma
        tracking_lin = jnp.exp(-jnp.square(body_lin_vel[0] - command[0]) / sigma)
        tracking_ang = jnp.exp(-jnp.square(ang_vel[2] - command[1]) / sigma)
        tilt = 1.0 - rot[2, 2]
        lin_vel_z = jnp.square(body_lin_vel[2])
        ang_vel_xy = jnp.sum(jnp.square(ang_vel[:2]))
        action_rate = jnp.sum(jnp.square(action - last_action))
        energy = jnp.sum(jnp.abs(data.actuator_force * data.qvel[_WHEEL_DOF_SLICE]))

        metrics["reward/tracking_lin_vel"] = tracking_lin
        metrics["reward/tracking_ang_vel"] = tracking_ang
        metrics["reward/tilt"] = -tilt

        scales = self._config.reward_config.scales
        return (
            scales.tracking_lin_vel * tracking_lin
            + scales.tracking_ang_vel * tracking_ang
            + scales.lin_vel_z * lin_vel_z
            + scales.ang_vel_xy * ang_vel_xy
            + scales.tilt * tilt
            + scales.action_rate * action_rate
            + scales.energy * energy
        )

    def _get_done(self, data: mjx.Data) -> jax.Array:
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        upside_down = rot[2, 2] < 0.4
        too_low = data.xpos[self._base_body_id, 2] < 0.03
        invalid = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        return (upside_down | too_low | invalid).astype(jnp.float32)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


class ObjectSpec(NamedTuple):
    keepout: float
    num_objects: int


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
    mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jnp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    return get_collision_info(state.contact, geom1, geom2)[0] < 0


def constrain_placement(placement: tuple, keepout: float) -> tuple:
    xmin, ymin, xmax, ymax = placement
    return xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout


def draw_placement(rng: jax.Array, keepout: float) -> jax.Array:
    xmin, ymin, xmax, ymax = constrain_placement(_NAV_EXTENTS, keepout)
    min_ = jnp.array([xmin, ymin])
    max_ = jnp.array([xmax, ymax])
    return jax.random.uniform(rng, shape=(2,), minval=min_, maxval=max_)


def placement_not_valid(
    xy: jax.Array,
    object_keepout: float,
    other_xy: jax.Array,
    other_keepout: jax.Array,
) -> jax.Array:
    def check_single(other_xy, other_keepout):
        dist = jnp.linalg.norm(xy - other_xy)
        return dist < (other_keepout + object_keepout)

    return jnp.any(jax.vmap(check_single)(other_xy, other_keepout))


def draw_until_valid(
    rng: jax.Array,
    object_keepout: float,
    other_xy: jax.Array,
    other_keepout: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def cond_fn(val):
        i, conflicted, *_ = val
        return jnp.logical_and(i < 10000, conflicted)

    def body_fn(val):
        i, _, _, rng = val
        rng, rng_ = jax.random.split(rng)
        xy = draw_placement(rng_, object_keepout)
        conflicted = placement_not_valid(xy, object_keepout, other_xy, other_keepout)
        return i + 1, conflicted, xy, rng

    init_val = (0, True, jnp.zeros((2,)), rng)
    i, _, xy, *_ = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return xy, i


def sample_navigation_layout(
    rng: jax.Array, objects_spec: dict[str, ObjectSpec]
) -> dict[str, list[tuple[int, jax.Array]]]:
    num_objects = sum(spec.num_objects for spec in objects_spec.values())
    all_placements = jnp.ones((num_objects, 2)) * 100.0
    all_keepouts = jnp.zeros(num_objects)
    layout = defaultdict(list)
    flat_idx = 0
    for name, object_spec in objects_spec.items():
        rng, rng_ = jax.random.split(rng)
        keys = jax.random.split(rng_, object_spec.num_objects)
        for key in keys:
            xy, _ = draw_until_valid(
                key, object_spec.keepout, all_placements, all_keepouts
            )
            all_placements = all_placements.at[flat_idx, :].set(xy)
            all_keepouts = all_keepouts.at[flat_idx].set(object_spec.keepout)
            layout[name].append((flat_idx, xy))
            flat_idx += 1
    return layout


def build_navigation_arena(
    spec: mujoco.MjSpec,
    layout: dict[str, list[tuple[int, jax.Array]]],
    *,
    goal_size: float,
    hazard_size: float,
    obstacle_size: float,
    visualize_lidar: bool = False,
):
    floor = spec.worldbody.geoms[0]
    assert floor.name == "floor"
    size = max(_NAV_EXTENTS)
    floor.size = jnp.array([size + 0.1, size + 0.1, 0.05])

    goal_pos = layout["goal"][0][1]
    goal = spec.worldbody.add_body(
        name="goal", mocap=True, pos=jnp.hstack([goal_pos, goal_size])
    )
    goal.add_geom(
        name="goal",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[goal_size, 0, 0],
        rgba=[0.0, 1.0, 0.0, 0.35],
        contype=jnp.zeros(()),
        conaffinity=jnp.zeros(()),
    )

    for i, (_, xy) in enumerate(layout["hazards"]):
        hazard = spec.worldbody.add_body(
            name=f"hazard_{i}",
            mocap=True,
            pos=jnp.hstack([xy, _HAZARD_BOX_HALF_HEIGHT]),
        )
        hazard.add_geom(
            name=f"hazard_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hazard_size, hazard_size, _HAZARD_BOX_HALF_HEIGHT],
            rgba=[0.1, 0.4, 1.0, 0.25],
            userdata=jnp.ones(1),
            contype=jnp.zeros(()),
            conaffinity=jnp.zeros(()),
        )

    for i, (_, xy) in enumerate(layout["obstacles"]):
        obstacle_height = obstacle_size
        collider_radius = obstacle_size * 0.35
        collider_half_length = obstacle_size - collider_radius
        obstacle = spec.worldbody.add_body(
            name=f"obstacle_{i}", pos=jnp.hstack([xy, obstacle_height])
        )
        obstacle.add_geom(
            name=f"obstacle_{i}_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[obstacle_size, obstacle_size, obstacle_height],
            rgba=[0.85, 0.25, 0.2, 1.0],
            contype=jnp.zeros(()),
            conaffinity=jnp.zeros(()),
        )
        obstacle.add_geom(
            name=f"obstacle_{i}_collider_x",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[collider_radius, collider_half_length, 0.0],
            quat=_CAPSULE_ROTATE_TO_X,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        obstacle.add_geom(
            name=f"obstacle_{i}_collider_y",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[collider_radius, collider_half_length, 0.0],
            quat=_CAPSULE_ROTATE_TO_Y,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )

    if visualize_lidar:
        robot_body = spec.body("base_link")
        for i, category in enumerate(lidar.LIDAR_GROUPS):
            lidar_body = robot_body.add_body(name=f"lidar_{category}")
            for bin_ in range(lidar.NUM_LIDAR_BINS):
                theta = 2 * jnp.pi * (bin_ + 0.5) / lidar.NUM_LIDAR_BINS
                binpos = jnp.array(
                    [
                        jnp.cos(theta) * lidar.RADIANS,
                        jnp.sin(theta) * lidar.RADIANS,
                        lidar.BASE_OFFSET + lidar.OFFSET_STEP * i,
                    ]
                )
                rgba = [0, 0, 0, 1]
                rgba[i] = 1
                lidar_body.add_site(
                    name=f"lidar_{category}_{bin_}",
                    size=lidar.LIDAR_SIZE * jnp.ones(3),
                    rgba=rgba,
                    pos=binpos,
                )


class JackalGoToGoal(MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = go_to_goal_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._xml_path = _XML_PATH.as_posix()
        self.goal_size = float(self._config.goal_size)
        self.hazard_size = float(self._config.hazard_size)
        self.obstacle_size = float(self._config.obstacle_size)
        self.spec = {
            "robot": ObjectSpec(float(self._config.robot_keepout), 1),
            "goal": ObjectSpec(float(self._config.goal_keepout), 1),
            "hazards": ObjectSpec(
                float(self._config.hazard_keepout), int(self._config.num_hazards)
            ),
            "obstacles": ObjectSpec(
                float(self._config.obstacle_keepout), int(self._config.num_obstacles)
            ),
        }
        mj_spec = mujoco.MjSpec.from_file(filename=self._xml_path, assets={})
        layout = sample_navigation_layout(jax.random.PRNGKey(self._config.seed), self.spec)
        build_navigation_arena(
            mj_spec,
            layout,
            goal_size=self.goal_size,
            hazard_size=self.hazard_size,
            obstacle_size=self.obstacle_size,
            visualize_lidar=bool(self._config.visualize_lidar),
        )
        self._mj_model = mj_spec.compile()
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self):
        self._base_body_id = self._mj_model.body("base_link").id
        self._chassis_geom_id = self._mj_model.geom("chassis_collision").id
        self._goal_mocap_id = self._mj_model.body("goal").mocapid[0]
        self._hazard_body_ids = [
            self._mj_model.body(f"hazard_{i}").id
            for i in range(self.spec["hazards"].num_objects)
        ]
        self._hazard_mocap_ids = [
            self._mj_model.body(f"hazard_{i}").mocapid[0]
            for i in range(self.spec["hazards"].num_objects)
        ]
        self._obstacle_body_ids = [
            self._mj_model.body(f"obstacle_{i}").id
            for i in range(self.spec["obstacles"].num_objects)
        ]
        self._obstacle_geom_ids = [
            self._mj_model.geom(f"obstacle_{i}_collider_x").id
            for i in range(self.spec["obstacles"].num_objects)
        ] + [
            self._mj_model.geom(f"obstacle_{i}_collider_y").id
            for i in range(self.spec["obstacles"].num_objects)
        ]
        robot_body_names = {
            "base_link",
            "chassis_link",
            "front_left_wheel_link",
            "front_right_wheel_link",
            "rear_left_wheel_link",
            "rear_right_wheel_link",
            "imu_link",
        }
        robot_body_ids = {
            self._mj_model.body(name).id for name in robot_body_names
        }
        self._robot_geom_ids = [
            i
            for i in range(self._mj_model.ngeom)
            if self._mj_model.geom_bodyid[i] in robot_body_ids and i != _FLOOR_GEOM_ID
        ]
        self._init_q = jnp.array(self._mj_model.qpos0)

    def robot_position(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._base_body_id]

    def goal_position(self, data: mjx.Data) -> jax.Array:
        return data.mocap_pos[self._goal_mocap_id]

    def goal_delta(self, data: mjx.Data) -> jax.Array:
        return self.goal_position(data) - self.robot_position(data)

    def goal_delta_body_frame(self, data: mjx.Data) -> jax.Array:
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        return rot.T @ self.goal_delta(data)

    def hazard_positions(self, data: mjx.Data) -> jax.Array:
        if not self._hazard_body_ids:
            return jnp.zeros((0, 3))
        return data.xpos[jnp.array(self._hazard_body_ids)]

    def obstacle_positions(self, data: mjx.Data) -> jax.Array:
        if not self._obstacle_body_ids:
            return jnp.zeros((0, 3))
        return data.xpos[jnp.array(self._obstacle_body_ids)]

    def _update_layout(self, layout, rng):
        qpos = self._init_q
        qvel = jnp.zeros(self.mjx_model.nv)
        mocap_pos = jnp.zeros((self.mjx_model.nmocap, 3))
        for name, items in layout.items():
            _, positions = zip(*items)
            if name == "goal":
                xyz = jnp.hstack([positions[0], self.goal_size])
                mocap_pos = mocap_pos.at[self._goal_mocap_id].set(xyz)
            elif name == "hazards":
                for pos, mocap_id in zip(positions, self._hazard_mocap_ids):
                    mocap_pos = mocap_pos.at[mocap_id].set(
                        jnp.hstack([pos, _HAZARD_BOX_HALF_HEIGHT])
                    )
            elif name == "robot":
                rng, rng_ = jax.random.split(rng)
                yaw = jax.random.uniform(rng_, (), minval=-jnp.pi, maxval=jnp.pi)
                cy = jnp.cos(yaw * 0.5)
                sy = jnp.sin(yaw * 0.5)
                qpos = qpos.at[:2].set(positions[0])
                qpos = qpos.at[3:7].set(jnp.array([cy, 0.0, 0.0, sy]))
            else:
                continue
        mocap_quat = jnp.zeros((self.mjx_model.nmocap, 4))
        mocap_quat = mocap_quat.at[:, 0].set(1.0)
        return qpos, qvel, mocap_pos, mocap_quat

    def reset(self, rng: jax.Array) -> State:
        layout = sample_navigation_layout(rng, self.spec)
        rng, rng_ = jax.random.split(rng)
        qpos, qvel, mocap_pos, mocap_quat = self._update_layout(layout, rng_)
        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros(self.mjx_model.nu),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )
        goal_dist = jnp.linalg.norm(self.goal_delta(data)[:2])
        info = {
            "rng": rng,
            "last_goal_dist": goal_dist,
            "goal_reached": jnp.zeros(()),
            "cost": jnp.zeros(()),
            "last_act": jnp.zeros(self.action_size),
            "collision": jnp.zeros(()),
            "hazard_cost": jnp.zeros(()),
        }
        metrics = {
            "reward/progress": jnp.zeros(()),
            "reward/goal_bonus": jnp.zeros(()),
            "cost/collision": jnp.zeros(()),
            "cost/hazard": jnp.zeros(()),
        }
        obs = self._get_obs(data, info)
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = self._action_to_ctrl(action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        reward, goal_dist, goal_reached = self._get_reward(data, state.info, state.metrics)
        collision_cost, hazard_cost = self._get_cost_terms(data)
        cost = jnp.maximum(collision_cost, hazard_cost)
        done = self._get_done(data)
        info = dict(state.info)
        info.update(
            last_goal_dist=goal_dist,
            goal_reached=goal_reached,
            cost=cost,
            last_act=action,
            collision=collision_cost,
            hazard_cost=hazard_cost,
        )
        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        world_lin_vel = mjx_env.get_sensor_data(
            self.mj_model, data, "base_link_subtreelinvel"
        )
        body_lin_vel = rot.T @ world_lin_vel
        imu_accel = mjx_env.get_sensor_data(self.mj_model, data, "imu_accel")
        gyro = mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro")
        obstacle_pos = self.obstacle_positions(data)
        hazard_pos = self.hazard_positions(data)
        obstacle_targets = jnp.concatenate([obstacle_pos, hazard_pos], axis=0)
        obstacle_lidar = lidar.compute_lidar(self.robot_position(data), rot, obstacle_targets)
        goal_lidar = lidar.compute_lidar(
            self.robot_position(data), rot, self.goal_position(data)[None, :]
        )
        return jnp.concatenate(
            [
                obstacle_lidar,
                goal_lidar,
                imu_accel,
                body_lin_vel[:2],
                gyro,
                data.qvel[_WHEEL_DOF_SLICE],
                info["last_act"],
            ]
        )

    def _get_reward(self, data: mjx.Data, info: dict[str, Any], metrics: dict[str, Any]):
        goal_dist = jnp.linalg.norm(self.goal_delta(data)[:2])
        progress = info["last_goal_dist"] - goal_dist
        goal_reached = (goal_dist <= self.goal_size + 0.05).astype(jnp.float32)
        reward = progress + goal_reached * self._config.goal_reward
        metrics["reward/progress"] = progress
        metrics["reward/goal_bonus"] = goal_reached * self._config.goal_reward
        return reward, goal_dist, goal_reached

    def _get_cost_terms(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        collision = jnp.array(
            [
                geoms_colliding(data, robot_geom, obstacle_geom)
                for robot_geom, obstacle_geom in product(
                    self._robot_geom_ids, self._obstacle_geom_ids
                )
            ]
        )
        collision_cost = (jnp.sum(collision) > 0).astype(jnp.float32)
        if self._hazard_body_ids:
            hazard_offset = jnp.abs(
                self.hazard_positions(data)[:, :2] - self.robot_position(data)[:2]
            )
            hazard_cost = jnp.all(hazard_offset <= self.hazard_size, axis=1).astype(
                jnp.float32
            )
            hazard_cost = jnp.any(hazard_cost).astype(jnp.float32)
        else:
            hazard_cost = jnp.zeros(())
        return collision_cost, hazard_cost

    def _get_done(self, data: mjx.Data) -> jax.Array:
        rot = data.xmat[self._base_body_id].reshape(3, 3)
        upside_down = rot[2, 2] < 0.4
        too_low = data.xpos[self._base_body_id, 2] < 0.03
        invalid = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        return (upside_down | too_low | invalid).astype(jnp.float32)

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        left = action[0] * self._config.max_torque * self._config.action_scale
        right = action[1] * self._config.max_torque * self._config.action_scale
        return jnp.array([left, right, left, right])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
