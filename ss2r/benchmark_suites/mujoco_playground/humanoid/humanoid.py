import functools

import jax
import jax.numpy as jnp
import mujoco
from brax.envs import Wrapper
from mujoco_playground import MjxEnv, State, dm_control_suite

# Humanoid 액추에이터 이름 → 인덱스 매핑 (좌우 대칭으로 같은 gear 오프셋 적용)
_name_to_id = {
    "right_hip_x": 3,
    "left_hip_x": 9,
    "right_hip_y": 5,
    "left_hip_y": 11,
    "right_hip_z": 4,
    "left_hip_z": 10,
    "left_knee": 12,
    "right_knee": 6,
}


def domain_randomization(cfg, sys, params=None, rng: jax.Array = None):
    """
    cartpole과 동일한 구조:
      - rng가 있으면: dist로 파라미터 샘플링 후 _apply_params 적용 (rand_dynamics)
      - params가 있으면: 주어진 params를 그대로 _apply_params 적용 (shift_dynamics)
      - 마지막에 in_axes + sys.tree_replace 한 번만 수행
    반환: (sys, in_axes)
    파라미터 벡터 순서: [friction, gear_hip_x, gear_hip_y, gear_hip_z, gear_knee] (길이 5)
    """

    # ---- rng 분기에서만 사용: 구간 [dr_low, dr_high]에서 균등 샘플링하는 분포 (cartpole의 dist) ----
    if rng is not None:
        # cfg는 flatten된 train_params/eval_params (gear_hip_x, gear_hip_y, gear_hip_z 등)
        dr_low = jnp.array(
            [
                cfg.friction[0],
                cfg.gear_hip.x[0],
                cfg.gear_hip.y[0],
                cfg.gear_hip.z[0],
                cfg.gear_knee[0],
            ]
        )
        dr_high = jnp.array(
            [
                cfg.friction[1],
                cfg.gear_hip.x[1],
                cfg.gear_hip.y[1],
                cfg.gear_hip.z[1],
                cfg.gear_knee[1],
            ]
        )
        dist = functools.partial(
            jax.random.uniform,
            shape=(dr_low.shape[0],),
            minval=dr_low,
            maxval=dr_high,
        )

    # ---- 공통: 파라미터 벡터 p를 받아 sys에 반영한 geom_friction, actuator_gear 반환 (cartpole의 _apply_params와 동일 역할) ----
    def _apply_params(p):
        """
        p = [friction, gear_hip_x, gear_hip_y, gear_hip_z, gear_knee]
        반환: (friction_sample, gear_sample) — tree_replace에 넣을 새 값
        """
        friction_val, gear_hip_x, gear_hip_y, gear_hip_z, gear_knee = (
            p[0], p[1], p[2], p[3], p[4]
        )

        # 마찰: 첫 번째 geom에만 오프셋 추가 후 [0, 1]로 클리핑
        friction_sample = sys.geom_friction.copy()
        friction_sample = friction_sample.at[0, 0].add(friction_val)
        friction_sample = jnp.clip(friction_sample, a_min=0.0, a_max=1.0)

        # 기어: 좌우 대칭으로 hip_x/y/z, knee 오프셋을 _name_to_id 인덱스에 반영
        gear_sample = sys.actuator_gear.copy()
        name_values = {
            "right_hip_x": gear_hip_x,
            "left_hip_x": gear_hip_x,
            "right_hip_y": gear_hip_y,
            "left_hip_y": gear_hip_y,
            "right_hip_z": gear_hip_z,
            "left_hip_z": gear_hip_z,
            "left_knee": gear_knee,
            "right_knee": gear_knee,
        }
        for name, gear_amount in name_values.items():
            actuator_id = _name_to_id[name]
            gear_sample = gear_sample.at[actuator_id, 0].add(gear_amount)

        return friction_sample, gear_sample

    # ---- params 전용: 주어진 벡터를 그대로 적용 (cartpole의 shift_dynamics) ----
    def shift_dynamics(params_vec):
        # params_vec shape: (5,) 또는 배치 시 (B, 5)
        friction_sample, gear_sample = _apply_params(params_vec)
        return friction_sample, gear_sample

    # ---- rng 전용: dist로 한 번 샘플링한 뒤 적용 (cartpole의 rand_dynamics) ----
    def rand_dynamics(rng_i):
        p = dist(rng_i)  # shape (5,)
        friction_sample, gear_sample = _apply_params(p)
        return friction_sample, gear_sample, p

    # ---- 분기: params / rng 둘 중 하나만 받음 (cartpole과 동일) ----
    if rng is None and params is not None:
        friction_sample, gear_sample = shift_dynamics(params)
    elif rng is not None and params is None:
        friction_sample, gear_sample, packed = rand_dynamics(rng)
    else:
        raise ValueError("Provide exactly one of (rng, params).")

    # ---- in_axes + sys.tree_replace 한 번만 수행 (cartpole과 동일 패턴) ----
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_friction": friction_sample,
            "actuator_gear": gear_sample,
        }
    )
    return sys, in_axes


def normalize_angle(angle, lower_bound=-jnp.pi, upper_bound=jnp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


class ConstraintWrapper(Wrapper):
    def __init__(self, env: MjxEnv, angle_tolerance: float):
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jnp.pi / 180.0
        joint_names = [
            "abdomen_z",
            "abdomen_y",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ]
        joint_ids = jnp.asarray(
            [
                mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)
                for name in joint_names
            ]
        )
        self.joint_ranges = [env.mj_model.jnt_range[id_] for id_ in joint_ids]
        self.qpos_ids = jnp.asarray(
            [env.mj_model.jnt_qposadr[id_] for id_ in joint_ids]
        )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        cost = jnp.zeros_like(nstate.reward)
        for qpos_id, joint_range in zip(self.qpos_ids, self.joint_ranges):
            angle = nstate.data.qpos[qpos_id]
            normalized_angle = normalize_angle(angle)
            lower_limit = normalize_angle(joint_range[0] - self.angle_tolerance)
            upper_limit = normalize_angle(joint_range[1] + self.angle_tolerance)
            is_out_of_range_case1 = (normalized_angle < lower_limit) & (
                normalized_angle >= upper_limit
            )
            is_out_of_range_case2 = (normalized_angle < lower_limit) | (
                normalized_angle >= upper_limit
            )
            out_of_range = jnp.where(
                upper_limit < lower_limit, is_out_of_range_case1, is_out_of_range_case2
            )
            cost += out_of_range
        nstate.info["cost"] = (cost > 0).astype(jnp.float32)
        return nstate


def make_safe(**kwargs):
    limit = kwargs["config"]["angle_tolerance"]
    env = dm_control_suite.load("HumanoidWalk", **kwargs)
    env = ConstraintWrapper(env, limit)
    return env


dm_control_suite.register_environment(
    "SafeHumanoidWalk", make_safe, dm_control_suite.humanoid.default_config
)
