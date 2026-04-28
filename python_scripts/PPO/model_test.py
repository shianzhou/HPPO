"""单智能体模型测试入口。

功能：
- 加载 `single_hppo_*.ckpt` 或指定的模型文件
- 只做推理，不做学习、不存经验
- 复用现有 Webots 环境接口，支持抓取和抬腿两段流程
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from python_scripts.PPO.checkpoint_utils import _ensure_dir, _latest_single_ckpt, load_single_model
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.Project_config import gps_goal, path_list
from python_scripts.Webots_interfaces import Environment


@dataclass
class TestSummary:
    total_episodes: int = 0
    grasp_success_count: int = 0
    lift_success_count: int = 0
    skipped_lift_count: int = 0
    accumulated_reward: float = 0.0


def _reset_env(env: Environment, wait_ms: int = 500) -> None:
    env.reset()
    env.wait(wait_ms)


def _safe_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


class SingleAgentModelTester:
    def __init__(
        self,
        model_path: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        max_episodes: int = 30,
        max_steps_per_phase: int = 80,
        reset_wait_ms: int = 500,
    ):
        self.model_path = model_path
        self.ckpt_dir = ckpt_dir or path_list["model_path_catch_PPO_h"]
        self.max_episodes = max_episodes
        self.max_steps_per_phase = max_steps_per_phase
        self.reset_wait_ms = reset_wait_ms

        self.agent = hppo(num_servos=6, node_num=19, env_information=None)
        self.summary = TestSummary()

    def _load_model(self) -> int:
        _ensure_dir(self.ckpt_dir)
        
        # 如果指定了具体模型路径，验证文件存在
        if self.model_path:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"❌ 指定的模型文件不存在: {self.model_path}")
            print(f"✅ [ModelTest] 已加载指定模型: {self.model_path}")
        else:
            # 未指定路径，检查目录里是否有模型
            selected_model, _ = _latest_single_ckpt(self.ckpt_dir)
            if not selected_model:
                raise FileNotFoundError(
                    f"❌ 目录中未找到保存的模型! \n"
                    f"   模型路径: {self.ckpt_dir}\n"
                    f"   请先运行训练代码保存模型 (single_hppo_*.ckpt)，再运行测试。"
                )
            print(f"✅ [ModelTest] 已加载目录最新模型: {selected_model}")
        
        episode_start = load_single_model(self.model_path, self.agent, self.ckpt_dir)
        return episode_start

    def _get_obs(self, env: Environment, step: int = 0):
        imgs = []
        obs_img, obs_tensor = env.get_img(step, imgs)
        robot_state = env.get_robot_state()
        return obs_img, obs_tensor, robot_state

    def _choose_action(self, obs_tensor, robot_state, episode_num: int):
        action_dict = self.agent.choose_action(
            episode_num=episode_num,
            obs=(obs_tensor, robot_state),
            x_graph=robot_state,
        )
        return action_dict

    def _run_grasp_phase(self, env: Environment, episode_num: int) -> bool:
        print("[ModelTest] 进入抓取测试阶段")
        _reset_env(env, self.reset_wait_ms)

        steps = 0
        total_reward = 0.0
        prev_shoulder_action = 0.0
        prev_arm_action = 0.0
        grasp_success = False

        obs_img, obs_tensor, robot_state = self._get_obs(env, steps)

        while steps < self.max_steps_per_phase:
            action_dict = self._choose_action(obs_tensor, robot_state, episode_num)

            discrete_action = _safe_numpy(action_dict["discrete_action"])
            continuous_action = _safe_numpy(action_dict["continuous_action"])

            d1 = int(discrete_action[1])
            d2 = int(discrete_action[2])
            cur_shoulder = float(continuous_action[0])
            cur_arm = float(continuous_action[1])
            masked_shoulder = prev_shoulder_action if d1 == 0 else cur_shoulder
            masked_arm = prev_arm_action if d2 == 0 else cur_arm

            gps1, gps2, gps3, gps4, _ = env.print_gps()
            img_name = f"model_test_grasp_{episode_num}_{steps}.png"

            next_state, reward, done, good, goal, count = env.step(
                robot_state,
                masked_shoulder,
                masked_arm,
                steps,
                0.0,
                gps1,
                gps2,
                gps3,
                gps4,
                img_name,
            )

            total_reward += float(reward)
            prev_shoulder_action = masked_shoulder
            prev_arm_action = masked_arm
            steps += 1

            next_obs_img, next_obs_tensor = env.get_img(steps, [])
            robot_state = env.get_robot_state()
            obs_tensor = next_obs_tensor
            obs_img = next_obs_img

            left_sensors = [
                env.darwin.get_touch_sensor_value("grasp_L1"),
                env.darwin.get_touch_sensor_value("grasp_L1_1"),
                env.darwin.get_touch_sensor_value("grasp_L1_2"),
            ]
            right_sensors = [
                env.darwin.get_touch_sensor_value("grasp_R1"),
                env.darwin.get_touch_sensor_value("grasp_R1_1"),
                env.darwin.get_touch_sensor_value("grasp_R1_2"),
            ]
            left_any = any(left_sensors)
            right_any = any(right_sensors)
            current_distance = self._current_distance(env)

            if left_any and right_any and current_distance <= 0.04:
                grasp_success = True
                print("[ModelTest] 抓取成功，进入下一阶段")
                break

            if done == 1:
                print(f"[ModelTest] 抓取阶段结束，done=1, steps={steps}")
                break

        self.summary.accumulated_reward += total_reward
        if grasp_success:
            self.summary.grasp_success_count += 1
        return grasp_success

    def _current_distance(self, env: Environment) -> float:
        gps1, _, _, _, _ = env.print_gps()
        if len(gps1) < 3:
            return float("inf")
        dx = gps_goal[0] - gps1[1]
        dy = gps_goal[1] - gps1[2]
        return math.sqrt(dx * dx + dy * dy)

    def _run_lift_phase(self, env: Environment, episode_num: int) -> bool:
        print("[ModelTest] 进入抬腿测试阶段")
        steps = 0
        total_reward = 0.0
        lift_success = False

        obs_img, obs_tensor, robot_state = self._get_obs(env, steps)

        last_action_upper = robot_state[12] if len(robot_state) > 12 else 0.0
        last_action_lower = robot_state[13] if len(robot_state) > 13 else 0.0
        last_action_ankle = robot_state[14] if len(robot_state) > 14 else 0.0

        while steps < self.max_steps_per_phase:
            action_dict = self._choose_action(obs_tensor, robot_state, episode_num)

            discrete_action = _safe_numpy(action_dict["discrete_action"])
            continuous_action = _safe_numpy(action_dict["continuous_action"])

            tai_discrete_action = discrete_action[3:6]
            tai_continuous_action = continuous_action[2:5]

            action_upper = float(tai_continuous_action[0])
            action_lower = float(tai_continuous_action[1])
            action_ankle = float(tai_continuous_action[2])

            discrete_upper = int(tai_discrete_action[0])
            discrete_lower = int(tai_discrete_action[1])
            discrete_ankle = int(tai_discrete_action[2])

            action_upper_exec = last_action_upper if discrete_upper == 0 else action_upper
            action_lower_exec = last_action_lower if discrete_lower == 0 else action_lower
            action_ankle_exec = last_action_ankle if discrete_ankle == 0 else action_ankle

            gps0, gps1, gps2, gps3, gps4 = env.print_gps()
            next_state, reward, done, good, goal, count = env.step2(
                robot_state,
                action_upper_exec,
                action_lower_exec,
                action_ankle_exec,
                steps,
                0.0,
                gps0,
                gps1,
                gps2,
                gps3,
                gps4,
            )

            total_reward += float(reward)
            last_action_upper = action_upper_exec
            last_action_lower = action_lower_exec
            last_action_ankle = action_ankle_exec
            steps += 1

            next_obs_img, next_obs_tensor = env.get_img(steps, [])
            robot_state = env.get_robot_state()
            obs_tensor = next_obs_tensor
            obs_img = next_obs_img

            if count == 1:
                lift_success = True
                print("[ModelTest] 抬腿阶段检测到有效执行")

            if done == 1:
                print(f"[ModelTest] 抬腿阶段结束，done=1, steps={steps}")
                break

        self.summary.accumulated_reward += total_reward
        if lift_success:
            self.summary.lift_success_count += 1
        return lift_success

    def run(self):
        self._load_model()
        env = Environment()
        _reset_env(env, self.reset_wait_ms)

        grasp_success = False

        for episode_num in range(self.max_episodes):
            self.summary.total_episodes += 1
            print("\n==============================")
            print(f"[ModelTest] Episode {episode_num}")
            print("==============================")

            _, obs_tensor, robot_state = self._get_obs(env, 0)
            decision_dict = self._choose_action(obs_tensor, robot_state, episode_num)
            decision = int(decision_dict["decision"])
            print(f"[ModelTest] 顶层决策: {decision} (0=抓取, 1=抬腿)")

            if decision == 0:
                grasp_success = self._run_grasp_phase(env, episode_num)
                if not grasp_success:
                    print("[ModelTest] 抓取未成功，跳过抬腿阶段")
                    self.summary.skipped_lift_count += 1
            else:
                if not grasp_success:
                    print("[ModelTest] 当前没有抓取成功状态，抬腿阶段跳过")
                    self.summary.skipped_lift_count += 1
                else:
                    self._run_lift_phase(env, episode_num)
                    grasp_success = False

            _reset_env(env, self.reset_wait_ms)

        print("\n[ModelTest] 测试完成")
        print(f"[ModelTest] episodes={self.summary.total_episodes}")
        print(f"[ModelTest] grasp_success={self.summary.grasp_success_count}")
        print(f"[ModelTest] lift_success={self.summary.lift_success_count}")
        print(f"[ModelTest] skipped_lift={self.summary.skipped_lift_count}")
        print(f"[ModelTest] accumulated_reward={self.summary.accumulated_reward:.4f}")

        return self.summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单智能体模型测试")
    parser.add_argument("--model-path", type=str, default=None, help="指定 checkpoint 路径，不传则自动加载目录最新模型")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="checkpoint 目录，默认使用配置中的抓取目录")
    parser.add_argument("--episodes", type=int, default=30, help="测试 episode 数")
    parser.add_argument("--max-steps", type=int, default=80, help="每个阶段最大步数")
    parser.add_argument("--reset-wait-ms", type=int, default=500, help="每次 reset 后等待时间")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    tester = SingleAgentModelTester(
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        max_episodes=args.episodes,
        max_steps_per_phase=args.max_steps,
        reset_wait_ms=args.reset_wait_ms,
    )
    tester.run()


if __name__ == "__main__":
    main()