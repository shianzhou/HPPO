import argparse
import glob
import json
import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch

from python_scripts.PPO.hppo_01 import HPPO
from python_scripts.PPO.hppo import HPPO as DecisionHPPO
from python_scripts.Project_config import Darwin_config, gps_goal1, path_list
from python_scripts.Webots_interfaces import Environment


LEFT_GRASP_SENSORS = ("grasp_L1", "grasp_L1_1", "grasp_L1_2")
RIGHT_GRASP_SENSORS = ("grasp_R1", "grasp_R1_1", "grasp_R1_2")
LEFT_FOOT_SENSORS = ("foot_L1", "foot_L2")
RIGHT_FOOT_SENSORS = ("foot_R1", "foot_R2")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _latest_catch_ckpt(dir_path: str) -> Optional[str]:
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None

    def _episode_number(file_path: str) -> int:
        match = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return max(files, key=_episode_number)


def _latest_tai_ckpt(dir_path: str) -> Optional[str]:
    files = glob.glob(os.path.join(dir_path, "tai_agent_*_*.ckpt"))
    if not files:
        return None

    def _episode_tuple(file_path: str) -> Tuple[int, int]:
        base = os.path.basename(file_path).replace(".ckpt", "")
        parts = base.split("_")
        if len(parts) < 4:
            return 0, 0
        try:
            return int(parts[-2]), int(parts[-1])
        except Exception:
            return 0, 0

    return max(files, key=_episode_tuple)


def _latest_decision_ckpt(dir_path: str) -> Optional[str]:
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return None

    def _episode_number(file_path: str) -> int:
        match = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    return max(files, key=_episode_number)


def _latest_model_recursive(base_dir: str, role_keywords: Tuple[str, ...]) -> Optional[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return None

    candidates: List[str] = []
    for ext in ("*.ckpt", "*.pt", "*.pth"):
        candidates.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))

    if not candidates:
        return None

    matched = [
        file_path
        for file_path in candidates
        if any(keyword in os.path.basename(file_path).lower() for keyword in role_keywords)
    ]
    pool = matched if matched else candidates

    def _number_key(file_path: str) -> Tuple[int, ...]:
        numbers = re.findall(r"\d+", os.path.basename(file_path))
        return tuple(int(n) for n in numbers) if numbers else (-1,)

    return max(pool, key=lambda p: (_number_key(p), os.path.getmtime(p)))


def _number_key_for_sort(file_path: str) -> Tuple[int, ...]:
    numbers = re.findall(r"\d+", os.path.basename(file_path))
    return tuple(int(n) for n in numbers) if numbers else (-1,)


def _extract_state_dict_from_checkpoint(checkpoint, preferred_key: str) -> Optional[dict]:
    if isinstance(checkpoint, dict):
        preferred = checkpoint.get(preferred_key)
        if isinstance(preferred, dict):
            return preferred

        for key in ("policy", "policy_tai", "state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        return None

    return checkpoint if isinstance(checkpoint, dict) else None


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict

    normalized = state_dict
    prefixes = ("module.", "policy.", "policy_tai.")
    for prefix in prefixes:
        keys = list(normalized.keys())
        if keys and all(str(key).startswith(prefix) for key in keys):
            normalized = {str(key)[len(prefix):]: value for key, value in normalized.items()}
    return normalized


def _is_state_dict_compatible(agent: HPPO, candidate_state_dict: dict) -> bool:
    model_state = agent.policy.state_dict()
    candidate_state_dict = _normalize_state_dict_keys(candidate_state_dict)

    for key, model_tensor in model_state.items():
        if key not in candidate_state_dict:
            return False
        if candidate_state_dict[key].shape != model_tensor.shape:
            return False
    return True


def _sorted_candidates(base_dir: str, patterns: Tuple[str, ...], keywords: Tuple[str, ...]) -> List[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return []

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base_dir, pattern)))

    files = list(dict.fromkeys(files))
    if keywords:
        files = [
            path
            for path in files
            if any(keyword in os.path.basename(path).lower() for keyword in keywords)
        ]

    files.sort(key=lambda p: (_number_key_for_sort(p), os.path.getmtime(p)), reverse=True)
    return files


def _pick_latest_compatible_model(
    agent: HPPO,
    preferred_key: str,
    preferred_dir: str,
    fallback_dir: str,
    patterns: Tuple[str, ...],
    keywords: Tuple[str, ...],
) -> Optional[str]:
    ordered_dirs = [preferred_dir]
    if fallback_dir and os.path.normpath(fallback_dir) != os.path.normpath(preferred_dir):
        ordered_dirs.append(fallback_dir)

    for base_dir in ordered_dirs:
        candidates = _sorted_candidates(base_dir, patterns=patterns, keywords=keywords)
        for candidate in candidates:
            try:
                checkpoint = torch.load(candidate, map_location="cpu")
                state_dict = _extract_state_dict_from_checkpoint(checkpoint, preferred_key)
                if state_dict is not None and _is_state_dict_compatible(agent, state_dict):
                    return candidate
            except Exception:
                continue
    return None


def _resolve_search_dir(model_path: Optional[str], default_dir: str) -> str:
    if not model_path:
        return default_dir

    normalized = os.path.normpath(model_path)
    if os.path.isdir(normalized):
        return normalized

    parent_dir = os.path.dirname(normalized)
    if parent_dir and os.path.isdir(parent_dir):
        return parent_dir

    return default_dir


def _resolve_model_path(
    input_path: Optional[str],
    default_dir: str,
    role_keywords: Tuple[str, ...],
    latest_fn,
) -> Optional[str]:
    if input_path is None:
        return latest_fn(default_dir) or _latest_model_recursive(default_dir, role_keywords)

    normalized = os.path.normpath(input_path)
    if os.path.isdir(normalized):
        return latest_fn(normalized) or _latest_model_recursive(normalized, role_keywords)

    return normalized


def _format_model_hint(search_dir: str, role_name: str) -> str:
    sample = []
    if search_dir and os.path.isdir(search_dir):
        for ext in ("*.ckpt", "*.pt", "*.pth"):
            sample.extend(glob.glob(os.path.join(search_dir, ext)))
    sample_names = ", ".join(os.path.basename(p) for p in sorted(sample)[:5]) if sample else "无"
    return (
        f"未找到{role_name}模型。已搜索目录: {search_dir}；目录内候选文件: {sample_names}。"
        f"请传入对应 *_model_path，或先完成训练生成模型后重试。"
    )


def _load_policy(agent: HPPO, ckpt_path: str, policy_key: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict_from_checkpoint(checkpoint, preferred_key=policy_key)
    if state_dict is None:
        raise RuntimeError(f"模型文件无法解析为 state_dict: {ckpt_path}")

    state_dict = _normalize_state_dict_keys(state_dict)
    model_state = agent.policy.state_dict()
    missing_keys = [key for key in model_state.keys() if key not in state_dict]
    mismatch_keys = [
        key
        for key in model_state.keys()
        if key in state_dict and state_dict[key].shape != model_state[key].shape
    ]
    if missing_keys or mismatch_keys:
        raise RuntimeError(
            f"模型与当前网络结构不匹配: {ckpt_path} | "
            f"missing={missing_keys[:6]} mismatch={mismatch_keys[:6]}"
        )

    filtered_state_dict = {key: state_dict[key] for key in model_state.keys()}
    try:
        agent.policy.load_state_dict(filtered_state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"模型与当前网络结构不匹配: {ckpt_path} | {exc}") from exc
    agent.policy.eval()


def _motor_index(env: Environment, motor_name: str) -> int:
    return env.darwin.motorName.index(motor_name)


def _set_motor(env: Environment, motor_name: str, position: float, velocity: float = 1.0) -> None:
    idx = _motor_index(env, motor_name)
    env.darwin.motors[idx].setVelocity(float(velocity))
    env.darwin.motors[idx].setPosition(float(position))


def _clip_with_joint_limit(joint_idx: int, value: float) -> float:
    low, high = Darwin_config.limit[joint_idx]
    return float(max(low, min(high, value)))


def _any_sensor_on(env: Environment, sensor_names: Tuple[str, ...]) -> bool:
    return any(env.get_touch_sensor_value(name) > 0.5 for name in sensor_names)


def _close_gripper(env: Environment, left: bool = False, right: bool = False) -> None:
    if left:
        _set_motor(env, "GraspL", -0.5, velocity=2.0)
    if right:
        _set_motor(env, "GraspR", -0.5, velocity=2.0)


def _open_gripper(env: Environment, left: bool = False, right: bool = False) -> None:
    if left:
        _set_motor(env, "GraspL", 1.0, velocity=2.0)
    if right:
        _set_motor(env, "GraspR", 1.0, velocity=2.0)


def _hand_targets_from_action(shoulder_action: float, arm_action: float) -> Tuple[float, float]:
    left_shoulder_target = _clip_with_joint_limit(1, 0.2995 * shoulder_action - 0.145)
    left_arm_target = _clip_with_joint_limit(5, 1.25 * arm_action + 0.25)
    return left_shoulder_target, left_arm_target


def _apply_left_hand_action(env: Environment, shoulder_action: float, arm_action: float) -> Tuple[float, float]:
    left_shoulder_target, left_arm_target = _hand_targets_from_action(shoulder_action, arm_action)
    _set_motor(env, "ShoulderL", left_shoulder_target)
    _set_motor(env, "ArmLowerL", left_arm_target)
    return left_shoulder_target, left_arm_target


def _hold_left_hand_pose(env: Environment, left_shoulder_target: float, left_arm_target: float) -> None:
    _set_motor(env, "ShoulderL", left_shoulder_target, velocity=0.4)
    _set_motor(env, "ArmLowerL", left_arm_target, velocity=0.4)


def _apply_right_hand_copy(env: Environment, left_shoulder_target: float, left_arm_target: float) -> None:
    right_shoulder_target = _clip_with_joint_limit(0, -left_shoulder_target)
    right_arm_target = _clip_with_joint_limit(4, -left_arm_target)
    _set_motor(env, "ShoulderR", right_shoulder_target)
    _set_motor(env, "ArmLowerR", right_arm_target)


def _apply_right_hand_copy_same(env: Environment, left_shoulder_target: float, left_arm_target: float) -> None:
    right_shoulder_target = _clip_with_joint_limit(0, left_shoulder_target)
    right_arm_target = _clip_with_joint_limit(4, left_arm_target)
    _set_motor(env, "ShoulderR", right_shoulder_target)
    _set_motor(env, "ArmLowerR", right_arm_target)


def _copy_right_arm_three_joints_from_left(env: Environment, velocity: float = 0.8) -> Tuple[float, float, float]:
    state = env.get_robot_state()
    left_shoulder = _clip_with_joint_limit(1, float(state[1]))
    left_arm_upper = _clip_with_joint_limit(3, float(state[3]))
    left_arm_lower = _clip_with_joint_limit(5, float(state[5]))

    right_shoulder = _clip_with_joint_limit(0, -left_shoulder)
    right_arm_upper = _clip_with_joint_limit(2, -left_arm_upper)
    right_arm_lower = _clip_with_joint_limit(4, -left_arm_lower)

    _set_motor(env, "ShoulderR", right_shoulder, velocity=velocity)
    _set_motor(env, "ArmUpperR", right_arm_upper, velocity=velocity)
    _set_motor(env, "ArmLowerR", right_arm_lower, velocity=velocity)
    return right_shoulder, right_arm_upper, right_arm_lower


def _hold_right_hand_pose_from_left(env: Environment, left_targets: Tuple[float, float], mode: str) -> None:
    left_shoulder_target, left_arm_target = left_targets
    if mode == "mirror":
        right_shoulder_target = _clip_with_joint_limit(0, -left_shoulder_target)
        right_arm_target = _clip_with_joint_limit(4, -left_arm_target)
    else:
        right_shoulder_target = _clip_with_joint_limit(0, left_shoulder_target)
        right_arm_target = _clip_with_joint_limit(4, left_arm_target)
    _set_motor(env, "ShoulderR", right_shoulder_target, velocity=0.4)
    _set_motor(env, "ArmLowerR", right_arm_target, velocity=0.4)


def _get_foot_height(env: Environment) -> float:
    gps_values = env.print_gps()
    foot_gps = gps_values[4]
    if isinstance(foot_gps, (list, tuple)) and len(foot_gps) >= 2:
        return float(foot_gps[1])
    return 0.0


def _left_foot_success(env: Environment) -> bool:
    gps_values = env.print_gps()
    foot_gps = gps_values[4]
    if not isinstance(foot_gps, (list, tuple)) or len(foot_gps) < 3:
        return False
    dx = float(gps_goal1[0]) - float(foot_gps[1])
    dy = float(gps_goal1[1]) - float(foot_gps[2])
    distance = math.sqrt(dx * dx + dy * dy)
    return _any_sensor_on(env, LEFT_FOOT_SENSORS) and distance <= 0.035


def _apply_left_leg_delta(env: Environment, du: float, dl: float, da: float) -> None:
    state = env.get_robot_state()
    target_upper = _clip_with_joint_limit(11, state[11] + du)
    target_lower = _clip_with_joint_limit(13, state[13] + dl)
    target_ankle = _clip_with_joint_limit(15, state[15] + da)
    _set_motor(env, "LegUpperL", target_upper)
    _set_motor(env, "LegLowerL", target_lower)
    _set_motor(env, "AnkleL", target_ankle)


def _apply_right_leg_copy_delta(env: Environment, du: float, dl: float, da: float) -> None:
    state = env.get_robot_state()
    target_upper = _clip_with_joint_limit(10, state[10] - du)
    target_lower = _clip_with_joint_limit(12, state[12] - dl)
    target_ankle = _clip_with_joint_limit(14, state[14] - da)
    _set_motor(env, "LegUpperR", target_upper)
    _set_motor(env, "LegLowerR", target_lower)
    _set_motor(env, "AnkleR", target_ankle)


def _run_left_hand_phase(
    env: Environment,
    catch_agent: HPPO,
    cycle_idx: int,
    max_steps: int,
    prev_exec: List[float],
    close_after_step: int,
    action_pause_ms: int,
) -> Tuple[bool, List[float], Tuple[float, float]]:
    imgs: List[np.ndarray] = []
    last_targets = (0.0, 0.0)
    contact_streak = 0
    _open_gripper(env, left=True)
    env.wait(80)
    for step in range(max_steps):
        _, obs_tensor = env.get_img(step, imgs)
        robot_state = env.get_robot_state()
        action_dict = catch_agent.choose_action(
            episode_num=cycle_idx,
            obs=[obs_tensor, robot_state],
            x_graph=robot_state,
        )
        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]

        shoulder_exec = prev_exec[0] if int(discrete_action[0]) == 0 else float(continuous_action[0])
        arm_exec = prev_exec[1] if int(discrete_action[1]) == 0 else float(continuous_action[1])
        prev_exec[0], prev_exec[1] = shoulder_exec, arm_exec

        print(
            f"[抓取阶段][回合 {step + 1}/{max_steps}] 离散动作={list(map(int, discrete_action))} "
            f"连续动作={[float(continuous_action[0]), float(continuous_action[1])]} "
            f"执行动作={[float(shoulder_exec), float(arm_exec)]}"
        )

        last_targets = _apply_left_hand_action(env, shoulder_exec, arm_exec)
        env.wait(180)
        env.wait(max(0, int(action_pause_ms)))

        if _any_sensor_on(env, LEFT_GRASP_SENSORS):
            contact_streak += 1
        else:
            contact_streak = 0

        if contact_streak >= 1:
            for _ in range(3):
                _hold_left_hand_pose(env, last_targets[0], last_targets[1])
                _close_gripper(env, left=True)
                env.wait(120)
                env.wait(max(0, int(action_pause_ms)))
                if _any_sensor_on(env, LEFT_GRASP_SENSORS):
                    imgs.clear()
                    return True, prev_exec, last_targets

    imgs.clear()
    return False, prev_exec, last_targets


def _run_right_hand_copy(
    env: Environment,
    left_targets: Tuple[float, float],
    catch_agent: Optional[HPPO] = None,
    cycle_idx: int = 0,
    max_steps: int = 8,
    close_after_step: int = 19,
    action_pause_ms: int = 200,
    explore_steps: int = 12,
) -> bool:
    _open_gripper(env, right=True)
    env.wait(100)
    contact_streak = 0
    for step in range(max_steps):
        right_targets = _copy_right_arm_three_joints_from_left(env, velocity=0.8)
        env.wait(220)
        env.wait(max(0, int(action_pause_ms)))
        right_sensor_values = [env.get_touch_sensor_value(name) for name in RIGHT_GRASP_SENSORS]
        touched = any(value > 0.5 for value in right_sensor_values)
        contact_streak = contact_streak + 1 if touched else 0
        print(
            f"[右手复制抓取][步 {step + 1}/{max_steps}] "
            f"右臂目标(ShoulderR,ArmUpperR,ArmLowerR)={list(map(float, right_targets))} (镜像取反) "
            f"传感器={right_sensor_values} 触碰={'是' if touched else '否'}"
        )

        if step >= max(0, int(close_after_step)) and contact_streak >= 1:
            print("右手检测到台阶接触，执行闭爪抓取。")
            _close_gripper(env, right=True)
            env.wait(140)
            env.wait(max(0, int(action_pause_ms)))

            right_sensor_values_after = [env.get_touch_sensor_value(name) for name in RIGHT_GRASP_SENSORS]
            if any(value > 0.5 for value in right_sensor_values_after):
                print("右手抓取成功：先触碰后闭爪。")
                return True

        _open_gripper(env, right=True)
        env.wait(80)

    if catch_agent is None:
        return False

    print("右手镜像阶段未触碰到台阶，切换为智能体自主探索右手动作。")
    prev_exec = [0.0, 0.0]
    imgs: List[np.ndarray] = []
    for step in range(max(1, int(explore_steps))):
        _, obs_tensor = env.get_img(step, imgs)
        robot_state = env.get_robot_state()
        action_dict = catch_agent.choose_action(
            episode_num=cycle_idx,
            obs=[obs_tensor, robot_state],
            x_graph=robot_state,
        )
        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]

        shoulder_exec = prev_exec[0] if int(discrete_action[0]) == 0 else float(continuous_action[0])
        arm_exec = prev_exec[1] if int(discrete_action[1]) == 0 else float(continuous_action[1])
        prev_exec[0], prev_exec[1] = shoulder_exec, arm_exec

        left_shoulder_target, left_arm_target = _hand_targets_from_action(shoulder_exec, arm_exec)
        right_shoulder_target = _clip_with_joint_limit(0, -left_shoulder_target)
        right_arm_target = _clip_with_joint_limit(4, -left_arm_target)

        _open_gripper(env, right=True)
        _set_motor(env, "ShoulderR", right_shoulder_target)
        _set_motor(env, "ArmLowerR", right_arm_target)
        env.wait(180)
        env.wait(max(0, int(action_pause_ms)))

        right_sensor_values = [env.get_touch_sensor_value(name) for name in RIGHT_GRASP_SENSORS]
        touched = any(value > 0.5 for value in right_sensor_values)
        print(
            f"[右手自主探索][步 {step + 1}/{max(1, int(explore_steps))}] "
            f"离散动作={list(map(int, discrete_action))} "
            f"连续动作={[float(continuous_action[0]), float(continuous_action[1])]} "
            f"执行动作={[float(shoulder_exec), float(arm_exec)]} "
            f"右臂目标(ShoulderR,ArmLowerR)={[float(right_shoulder_target), float(right_arm_target)]} "
            f"传感器={right_sensor_values} 触碰={'是' if touched else '否'}"
        )

        if touched and step >= max(0, int(close_after_step)):
            print("右手自主探索检测到接触，执行闭爪抓取。")
            _close_gripper(env, right=True)
            env.wait(140)
            env.wait(max(0, int(action_pause_ms)))
            right_sensor_values_after = [env.get_touch_sensor_value(name) for name in RIGHT_GRASP_SENSORS]
            if any(value > 0.5 for value in right_sensor_values_after):
                print("右手抓取成功：自主探索触碰后闭爪。")
                imgs.clear()
                return True

    imgs.clear()
    return False


def _run_left_leg_phase(
    env: Environment,
    tai_agent: HPPO,
    cycle_idx: int,
    max_steps: int,
    prev_exec: List[float],
    action_pause_ms: int,
) -> Tuple[bool, List[float], List[Tuple[float, float, float]]]:
    imgs: List[np.ndarray] = []
    executed_sequence: List[Tuple[float, float, float]] = []
    for step in range(max_steps):
        _, obs_tensor = env.get_img(step, imgs)
        robot_state = env.get_robot_state()
        action_dict = tai_agent.choose_action(
            episode_num=cycle_idx,
            obs=[obs_tensor, robot_state],
            x_graph=robot_state,
        )
        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]

        du = prev_exec[0] if int(discrete_action[0]) == 0 else float(continuous_action[0])
        dl = prev_exec[1] if int(discrete_action[1]) == 0 else float(continuous_action[1])
        da = prev_exec[2] if int(discrete_action[2]) == 0 else float(continuous_action[2])
        prev_exec[0], prev_exec[1], prev_exec[2] = du, dl, da

        print(
            f"[抬腿阶段][回合 {step + 1}/{max_steps}] 离散动作={list(map(int, discrete_action))} "
            f"连续动作={[float(continuous_action[0]), float(continuous_action[1]), float(continuous_action[2])]} "
            f"执行动作={[float(du), float(dl), float(da)]}"
        )

        executed_sequence.append((du, dl, da))
        _apply_left_leg_delta(env, du, dl, da)
        env.wait(200)
        env.wait(max(0, int(action_pause_ms)))

        if _left_foot_success(env):
            imgs.clear()
            return True, prev_exec, executed_sequence

    imgs.clear()
    return False, prev_exec, executed_sequence


def _run_right_leg_copy(
    env: Environment,
    executed_sequence: List[Tuple[float, float, float]],
    action_pause_ms: int,
    tai_agent: Optional[HPPO] = None,
    cycle_idx: int = 0,
    explore_steps: int = 12,
) -> bool:
    if not executed_sequence:
        return False

    for du, dl, da in executed_sequence:
        _apply_right_leg_copy_delta(env, du, dl, da)
        env.wait(200)
        env.wait(max(0, int(action_pause_ms)))
        if _any_sensor_on(env, RIGHT_FOOT_SENSORS):
            return True

    du, dl, da = executed_sequence[-1]
    for _ in range(6):
        _apply_right_leg_copy_delta(env, du, dl, da)
        env.wait(200)
        env.wait(max(0, int(action_pause_ms)))
        if _any_sensor_on(env, RIGHT_FOOT_SENSORS):
            return True

    if tai_agent is None:
        return False

    print("右脚镜像踩踏未成功，切换为智能体自主探索右脚动作。")
    prev_exec = [0.0, 0.0, 0.0]
    imgs: List[np.ndarray] = []
    for step in range(max(1, int(explore_steps))):
        _, obs_tensor = env.get_img(step, imgs)
        robot_state = env.get_robot_state()
        action_dict = tai_agent.choose_action(
            episode_num=cycle_idx,
            obs=[obs_tensor, robot_state],
            x_graph=robot_state,
        )
        discrete_action = action_dict["discrete_action"]
        continuous_action = action_dict["continuous_action"]

        du = prev_exec[0] if int(discrete_action[0]) == 0 else float(continuous_action[0])
        dl = prev_exec[1] if int(discrete_action[1]) == 0 else float(continuous_action[1])
        da = prev_exec[2] if int(discrete_action[2]) == 0 else float(continuous_action[2])
        prev_exec[0], prev_exec[1], prev_exec[2] = du, dl, da

        _apply_right_leg_copy_delta(env, du, dl, da)
        env.wait(200)
        env.wait(max(0, int(action_pause_ms)))

        right_sensor_values = [env.get_touch_sensor_value(name) for name in RIGHT_FOOT_SENSORS]
        touched = any(value > 0.5 for value in right_sensor_values)
        print(
            f"[右脚自主探索][步 {step + 1}/{max(1, int(explore_steps))}] "
            f"离散动作={list(map(int, discrete_action))} "
            f"连续动作={[float(continuous_action[0]), float(continuous_action[1]), float(continuous_action[2])]} "
            f"执行动作={[float(du), float(dl), float(da)]} "
            f"传感器={right_sensor_values} 触碰={'是' if touched else '否'}"
        )
        if touched:
            imgs.clear()
            return True

    imgs.clear()
    return False


def run_climb_test(
    robot,
    catch_model_path: Optional[str] = None,
    tai_model_path: Optional[str] = None,
    decision_model_path: Optional[str] = None,
    total_trials: int = 100,
    reset_settle_ms: int = 3000,
    action_pause_ms: int = 250,
    hand_close_after_step: int = 0,
    max_cycles: int = 1,
    max_hand_steps: int = 20,
    max_leg_steps: int = 20,
    level_height: float = 0.058,
    result_file: Optional[str] = None,
):
    env = Environment()
    env.reset()
    env.wait(max(0, int(reset_settle_ms)))

    catch_agent = HPPO(num_servos=2, node_num=19, env_information=None)
    tai_agent = HPPO(num_servos=3, node_num=19, env_information=None)
    decision_agent = DecisionHPPO(num_servos=1, node_num=19, env_information=None)

    catch_dir = path_list["model_path_catch_PPO_h"]
    tai_dir = path_list["model_path_tai_PPO_h"]
    decision_dir = path_list.get("model_path_decision_PPO_h", "")
    ppo_root = os.path.dirname(catch_dir)

    catch_search_dir = _resolve_search_dir(catch_model_path, catch_dir)
    tai_search_dir = _resolve_search_dir(tai_model_path, tai_dir)
    decision_search_dir = _resolve_search_dir(decision_model_path, decision_dir)

    if catch_model_path is not None and os.path.isfile(catch_model_path):
        catch_model_path = os.path.normpath(catch_model_path)
    else:
        catch_model_path = _pick_latest_compatible_model(
            agent=catch_agent,
            preferred_key="policy",
            preferred_dir=catch_search_dir,
            fallback_dir=catch_dir,
            patterns=("catch_hppo_*.ckpt", "*.ckpt", "*.pt", "*.pth"),
            keywords=("catch", "grasp"),
        )

    if tai_model_path is not None and os.path.isfile(tai_model_path):
        tai_model_path = os.path.normpath(tai_model_path)
    else:
        tai_model_path = _pick_latest_compatible_model(
            agent=tai_agent,
            preferred_key="policy_tai",
            preferred_dir=tai_search_dir,
            fallback_dir=tai_dir,
            patterns=("tai_agent_*_*.ckpt", "*.ckpt", "*.pt", "*.pth"),
            keywords=("tai", "leg"),
        )

    if decision_model_path is not None and os.path.isfile(decision_model_path):
        decision_model_path = os.path.normpath(decision_model_path)
    else:
        decision_model_path = _pick_latest_compatible_model(
            agent=decision_agent,
            preferred_key="policy",
            preferred_dir=decision_search_dir,
            fallback_dir=decision_dir,
            patterns=("decision_hppo_*.ckpt", "*.ckpt", "*.pt", "*.pth"),
            keywords=("decision", "hppo"),
        ) or _latest_decision_ckpt(decision_dir)

    if not catch_model_path or not os.path.exists(catch_model_path):
        raise FileNotFoundError(_format_model_hint(catch_dir, "抓取"))
    if not tai_model_path or not os.path.exists(tai_model_path):
        raise FileNotFoundError(_format_model_hint(tai_dir, "抬腿"))
    if not decision_model_path or not os.path.exists(decision_model_path):
        raise FileNotFoundError(_format_model_hint(decision_dir, "决策"))

    _load_policy(catch_agent, catch_model_path, policy_key="policy")
    _load_policy(tai_agent, tai_model_path, policy_key="policy_tai")
    _load_policy(decision_agent, decision_model_path, policy_key="policy")

    print(f"加载抓取模型: {catch_model_path}")
    print(f"加载抬腿模型: {tai_model_path}")
    print(f"加载决策模型: {decision_model_path}")

    total_trials = max(1, int(total_trials))
    total_decision_count = 0
    total_decision_correct = 0
    total_grasp_attempts = 0
    total_grasp_successes = 0
    total_step_attempts = 0
    total_step_successes = 0
    max_cycles = max(1, int(max_cycles))
    total_cycles = int(total_trials * max_cycles)
    executed_cycles = 0
    completed_cycle_successes = 0

    for trial in range(1, total_trials + 1):
        print(f"\n================ 测试 {trial}/{total_trials} ================")
        env.reset()
        env.wait(max(0, int(reset_settle_ms)))

        _close_gripper(env, left=False, right=False)
        _open_gripper(env, left=True, right=True)
        env.wait(300)

        prev_hand_exec = [0.0, 0.0]
        prev_leg_exec = [0.0, 0.0, 0.0]
        test_failed = False

        for climb_round in range(1, max_cycles + 1):
            executed_cycles += 1

            d_imgs: List[np.ndarray] = []
            _, d_obs_tensor = env.get_img(trial + climb_round, d_imgs)
            d_robot_state = env.get_robot_state()
            d_obs = (d_obs_tensor, d_robot_state)
            decision_dict = decision_agent.choose_action(obs=d_obs, x_graph=d_robot_state)
            decision = int(decision_dict["discrete_action"][0])
            total_decision_count += 1
            if decision == 0:
                total_decision_correct += 1
            print(f"[决策层-抓取前][测试 {trial}] decision={decision} (期望=0 抓取)")
            if decision != 0:
                print("决策错误：抓取前未选择抓取，本次测试结束。")
                test_failed = True
                break

            total_grasp_attempts += 1
            left_grasp_ok, prev_hand_exec, left_targets = _run_left_hand_phase(
                env=env,
                catch_agent=catch_agent,
                cycle_idx=trial + climb_round,
                max_steps=max_hand_steps,
                prev_exec=prev_hand_exec,
                close_after_step=max(0, min(hand_close_after_step, max_hand_steps - 1)),
                action_pause_ms=action_pause_ms,
            )
            if not left_grasp_ok:
                print("左手抓梯失败，本次测试结束。")
                test_failed = True
                break

            right_grasp_ok = _run_right_hand_copy(
                env,
                left_targets=left_targets,
                catch_agent=catch_agent,
                cycle_idx=trial + climb_round,
                max_steps=max_hand_steps,
                close_after_step=max(0, min(hand_close_after_step, max_hand_steps - 1)),
                action_pause_ms=action_pause_ms,
                explore_steps=max_hand_steps,
            )
            if not right_grasp_ok:
                print("右手抓梯失败，本次测试结束。")
                test_failed = True
                break
            total_grasp_successes += 1

            d_imgs2: List[np.ndarray] = []
            _, d_obs_tensor2 = env.get_img(trial + total_trials + climb_round, d_imgs2)
            d_robot_state2 = env.get_robot_state()
            decision_dict2 = decision_agent.choose_action(obs=(d_obs_tensor2, d_robot_state2), x_graph=d_robot_state2)
            decision2 = int(decision_dict2["discrete_action"][0])
            total_decision_count += 1
            if decision2 == 1:
                total_decision_correct += 1
            print(f"[决策层-踩踏前][测试 {trial}] decision={decision2} (期望=1 踩踏)")
            if decision2 != 1:
                print("决策错误：踩踏前未选择踩踏，本次测试结束。")
                test_failed = True
                break

            total_step_attempts += 1
            left_leg_ok, prev_leg_exec, executed_leg_sequence = _run_left_leg_phase(
                env=env,
                tai_agent=tai_agent,
                cycle_idx=trial + climb_round,
                max_steps=max_leg_steps,
                prev_exec=prev_leg_exec,
                action_pause_ms=action_pause_ms,
            )
            if not left_leg_ok:
                print("左脚抬腿踩阶失败，本次测试结束。")
                test_failed = True
                break

            right_leg_ok = _run_right_leg_copy(
                env,
                executed_sequence=executed_leg_sequence,
                action_pause_ms=action_pause_ms,
                tai_agent=tai_agent,
                cycle_idx=trial + climb_round,
                explore_steps=max_leg_steps,
            )
            if not right_leg_ok:
                print("右脚踩踏失败（镜像+自主探索均未成功），本次测试结束。")
                test_failed = True
                break

            total_step_successes += 1
            completed_cycle_successes += 1
            print(f"测试{trial}当前爬升成功，继续向上。")

            _open_gripper(env, left=True)
            _close_gripper(env, right=True)
            env.wait(250)

        if test_failed:
            continue

    result = {
        "configured_total_cycles": int(total_cycles),
        "executed_cycles": int(executed_cycles),
        "completed_cycle_successes": int(completed_cycle_successes),
        "completed_cycle_success_rate": float(completed_cycle_successes / executed_cycles) if executed_cycles else 0.0,
        "decision_total": int(total_decision_count),
        "decision_correct": int(total_decision_correct),
        "decision_success_rate": float(total_decision_correct / total_decision_count)
        if total_decision_count
        else 0.0,
        "grasp_attempts": int(total_grasp_attempts),
        "grasp_successes": int(total_grasp_successes),
        "grasp_success_rate": float(total_grasp_successes / total_grasp_attempts)
        if total_grasp_attempts
        else 0.0,
        "step_attempts": int(total_step_attempts),
        "step_successes": int(total_step_successes),
        "step_success_rate": float(total_step_successes / total_step_attempts)
        if total_step_attempts
        else 0.0,
    }

    if result_file is None:
        result_file = os.path.join(path_list["tai_log_path_PPO"], "climb_test_result.json")
    result_dir = os.path.dirname(result_file)
    if result_dir:
        _ensure_dir(result_dir)
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n测试完成。")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存到: {result_file}")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="测试训练好的PPO爬梯模型（单侧动作+另一侧复制）")
    parser.add_argument("--catch-model", type=str, default=None, help="抓取模型路径(.ckpt)")
    parser.add_argument("--tai-model", type=str, default=None, help="抬腿模型路径(.ckpt)")
    parser.add_argument("--decision-model", type=str, default=None, help="决策模型路径(.ckpt)")
    parser.add_argument("--total-trials", type=int, default=100, help="测试次数（默认100次，每次从决策开始完整执行一次）")
    parser.add_argument("--reset-settle-ms", type=int, default=3000, help="每次重置后缓冲时间（毫秒）")
    parser.add_argument("--action-pause-ms", type=int, default=250, help="每回合动作执行后的额外缓冲时间（毫秒）")
    parser.add_argument("--hand-close-after-step", type=int, default=0, help="每回合抓手允许闭合的最早步数")
    parser.add_argument("--max-cycles", type=int, default=20, help="每次测试最多连续上爬轮次（双脚成功后继续上爬）")
    parser.add_argument("--max-hand-steps", type=int, default=20, help="每轮左手尝试步数")
    parser.add_argument("--max-leg-steps", type=int, default=20, help="每轮左脚尝试步数")
    parser.add_argument("--level-height", type=float, default=0.058, help="单层台阶高度估计")
    parser.add_argument("--result-file", type=str, default=None, help="结果输出JSON路径")
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        from controllers import Robot
    except Exception as e:
        raise RuntimeError("该脚本需要在Webots控制器环境运行。") from e

    robot = Robot()
    run_climb_test(
        robot=robot,
        catch_model_path=args.catch_model,
        tai_model_path=args.tai_model,
        decision_model_path=args.decision_model,
        total_trials=args.total_trials,
        reset_settle_ms=args.reset_settle_ms,
        action_pause_ms=args.action_pause_ms,
        hand_close_after_step=args.hand_close_after_step,
        max_cycles=args.max_cycles,
        max_hand_steps=args.max_hand_steps,
        max_leg_steps=args.max_leg_steps,
        level_height=args.level_height,
        result_file=args.result_file,
    )


if __name__ == "__main__":
    main()