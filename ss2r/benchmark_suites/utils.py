from typing import Any

try:
    from omegaconf import DictConfig
except ModuleNotFoundError:
    DictConfig = Any


def get_domain_name(cfg: DictConfig) -> str:
    return cfg.environment.domain_name


def get_task_config(cfg: DictConfig) -> DictConfig:
    return cfg.environment


def flatten_params(params, prefix=""): 
    """
    이중 dict인 train_params/eval_params를 flatten하여 [low, high] 배열로 사용할 수 있게 만드는 헬퍼 함수
      - 값이 dict면 재귀, 키는 prefix_key 형태 (예: gear_hip_x).
      - 값이 [low, high]면 그대로 저장.

    ex) Humanoid :
    input : train_params = { friction: [0,0], gear_hip: { x: [-20,20], y: [-20,20], z: [-60,60] }, gear_knee: [0,0] }
    output: { "friction": [0, 0], "gear_hip_x": [-20, 20], "gear_hip_y": [-20, 20], "gear_hip_z": [-60, 60], "gear_knee": [0, 0] }    
    """
    result = {}
    for k in params.keys():
        v = params[k]
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            sub = flatten_params(v, prefix=key)
            for k2 in sub.keys():
                result[k2] = sub[k2]
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            result[key] = [float(v[0]), float(v[1])]
        else:
            raise TypeError(
                f"param '{key}': expected [low, high] or nested dict, got {type(v)}"
            )
    return result
