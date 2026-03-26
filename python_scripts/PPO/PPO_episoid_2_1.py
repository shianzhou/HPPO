import math
import os
import numpy as np
import torch
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import gps_goal1,path_list
from python_scripts.PPO.hppo_01 import HPPO as tai_hppo
#from python_scripts.PPO.utils.sensor_utils import wait_for_sensors_stable, reset_environment

# 统一的模型保存目录（仅使用配置路径）
tai_checkpoint_dir = path_list['model_path_tai_PPO_h']
os.makedirs(tai_checkpoint_dir, exist_ok=True)
def validate_and_clean_data(data, default_value=0.0):
    """验证并清理数据中的NaN和Inf值"""
    if isinstance(data, (list, tuple)):
        return [validate_and_clean_data(x, default_value) for x in data]
    elif isinstance(data, np.ndarray):
        cleaned = np.nan_to_num(data, nan=default_value, posinf=default_value, neginf=-default_value)
        return cleaned
    elif isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return default_value
        return data
    else:
        return data
def PPO_tai_episoid(existing_env=None ,total_episode=0, episode=0, log_writer_tai=None, log_file_latest_tai=None, catch_success=False, tai_agent=None, training_manager=None):

    # 如果没有抓取成功，直接跳过抬腿阶段
    if not catch_success:
        print("⚠️ 未检测到抓取成功，跳过抬腿阶段。")
        return

    # 智能体实例化（优先使用外部传入的模型）
    if tai_agent is None:
        tai_agent = tai_hppo(num_servos=3, node_num=19, env_information=None)

    # 使用已有的环境实例或创建新的
    if existing_env is not None:
        env = existing_env
    else:
        env = Environment()

    print("开始抬腿！")
    env.darwin.tai_leg_L1()
    env.darwin.tai_leg_L2()
        
    # 初始化状态
    count = 0
    return_all = 0
    prev_distance = None  # 用于计算距离变化奖励
    prev_foot_height = None
    imgs = []
    goal = 0
    done = 0
    reward = 0
    steps = 0
    catch_flag = 0
    gate_activation = {"upper": 0.0, "lower": 0.0, "ankle": 0.0, "all_off": 0, "steps": 0}

    # 初始化用于保持姿势的变量
    robot_state_initial = env.get_robot_state()
    last_action_LegUpper = robot_state_initial[12] if len(robot_state_initial) > 0 else 0.0  # LAnklePitch
    last_action_LegLower = robot_state_initial[13] if len(robot_state_initial) > 1 else 0.0 # LAnkleRoll
    last_action_Ankle = robot_state_initial[14] if len(robot_state_initial) > 2 else 0.0    # LAnkleRoll

    # 获取观察和状态
    print("____________________")
   

    # 记录回合数
    log_writer_tai.add(episode_num=total_episode)
    
    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        robot_state = validate_and_clean_data(robot_state)
        # 将机器人状态转换为PPO状态，与PPO_episoid_1.py保持一致
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]
        # 选择动作 - 使用tai_agent替代三个独立的PPO2智能体

        tai_dict = tai_agent.choose_action(episode_num=episode,
                                       obs=[obs_img, robot_state],
                                       x_graph=robot_state)

        # 从tai_agent获取离散和连续动作
        tai_discrete_action = tai_dict['discrete_action']  # [3] - 三个舵机的离散动作
        tai_continuous_action = tai_dict['continuous_action']  # [3] - 三个舵机的连续动作
        tai_continuous_log_prob = tai_dict['continuous_log_prob']  # [3] - 连续动作的对数概率
        tai_value = tai_dict['value']  # 状态价值

        # 提取三个关节的连续动作和对应的离散动作
        action_LegUpper = float(tai_continuous_action[0])  # 上腿关节
        action_LegLower = float(tai_continuous_action[1])  # 下腿关节
        action_Ankle = float(tai_continuous_action[2])  # 踝关节

        # 提取离散动作（门控信号）
        discrete_upper = int(tai_discrete_action[0])
        discrete_lower = int(tai_discrete_action[1])
        discrete_ankle = int(tai_discrete_action[2])

        # 提取对应的对数概率
        log_prob_LegUpper = float(tai_continuous_log_prob[0])
        log_prob_LegLower = float(tai_continuous_log_prob[1])
        log_prob_Ankle = float(tai_continuous_log_prob[2])

        action_LegUpper_original = action_LegUpper
        action_LegLower_original = action_LegLower
        action_Ankle_original = action_Ankle

        # 根据离散动作决定是使用新动作还是保持上一动作
        if discrete_upper == 0:
            action_LegUpper_exec = last_action_LegUpper
        else:
            action_LegUpper_exec = action_LegUpper

        if discrete_lower == 0:
            action_LegLower_exec = last_action_LegLower
        else:
            action_LegLower_exec = action_LegLower

        if discrete_ankle == 0:
            action_Ankle_exec = last_action_Ankle
        else:
            action_Ankle_exec = action_Ankle

        # 更新 "last_action" 变量以备下一步使用
        last_action_LegUpper = action_LegUpper_exec
        last_action_LegLower = action_LegLower_exec
        last_action_Ankle = action_Ankle_exec

        gate_activation["steps"] += 1
        gate_activation["upper"] += discrete_upper
        gate_activation["lower"] += discrete_lower
        gate_activation["ankle"] += discrete_ankle
        
        # 分别添加动作、对数概率和状态价值到日志
        log_writer_tai.add_action_tai(action_LegUpper, action_LegLower, action_Ankle)
        log_writer_tai.add_log_prob_tai(log_prob_LegUpper, log_prob_LegLower, log_prob_Ankle)
        value_scalar = tai_value if isinstance(tai_value, (int, float)) else tai_value.item()
        log_writer_tai.add_value_tai(value_scalar, value_scalar, value_scalar)

        print("第", steps + 1, "步")
        print(f"【tai_agent门控控制】离散动作: [{discrete_upper}, {discrete_lower}, {discrete_ankle}]")
        print(f"【原始连续动作】LegUpper: {action_LegUpper_original:.4f}, LegLower: {action_LegLower_original:.4f}, Ankle: {action_Ankle_original:.4f}")
        print(f"【最终执行动作】LegUpper: {action_LegUpper_exec:.4f}, LegLower: {action_LegLower_exec:.4f}, Ankle: {action_Ankle_exec:.4f}")
        
        # 详细的关节启用状态调试信息
        print(f"【关节启用状态】")
        print(f"  LegUpper: {'启用' if discrete_upper == 1 else '关闭'} -> 角度={action_LegUpper_exec:.4f}")
        print(f"  LegLower: {'启用' if discrete_lower == 1 else '关闭'} -> 角度={action_LegLower_exec:.4f}")
        print(f"  Ankle:    {'启用' if discrete_ankle == 1 else '关闭'} -> 角度={action_Ankle_exec:.4f}")

        # 获取GPS数据
        gps_values = env.print_gps()
        gps_values = validate_and_clean_data(gps_values)
            
        # 设置抓取器状态
        catch_flag = 0.0
            
        # 生成图像名称
        img_name = f"img{steps}.png"
            
        # 执行动作
        next_state, reward_env, done, good, goal, count = env.step2(
            robot_state, action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec, steps, catch_flag, 
            gps_values[4], gps_values[0], gps_values[1], gps_values[2], gps_values[3],
        )
        
        # ========== 奖励计算（统一使用自定义reward） ==========
        reward = 0.0
        if count == 1:
            x1 = gps_goal1[0] - gps_values[4][1]  # 目标x坐标 - 当前脚部GPS x坐标
            y1 = gps_goal1[1] - gps_values[4][2]  # 目标y坐标 - 当前脚部GPS y坐标
            distance = math.sqrt(x1 * x1 + y1 * y1)  # 计算欧几里得距离
            if prev_distance is not None:
                reward += (prev_distance - distance) * 10.0  # 靠近目标得到正奖励
            else:
                reward -= distance  # 初始步鼓励靠近
            prev_distance = distance

            # 检查水平位置是否正确 (XY平面)
            is_position_correct = distance <= 0.06  # 6cm以内视为位置正确

            if is_position_correct:
                reward += 5  # 位置正确奖励

            foot_height = float(gps_values[4][0])  # 【修正】使用Z轴作为高度
            if prev_foot_height is not None:
                height_diff = foot_height - prev_foot_height

                if episode < 5:
                    # 前5个回合，鼓励抬高
                    if height_diff > 0:
                        reward += height_diff * 5.0  # 增加权重鼓励抬高
                else:
                    # 5个回合后，如果位置正确，则鼓励踩踏
                    if is_position_correct and height_diff < 0:
                        # 位置正确，且正在下降，给予大量踩踏奖励
                        reward += -height_diff * 20.0
                        print("✅ 位置正确，鼓励踩踏！")
                    elif height_diff < 0:
                        # 位置不正确，但仍在下降，给予少量奖励
                        reward += -height_diff * 3.0
                    else:
                        # 任何时候抬高都给予惩罚（5回合后）
                        reward -= height_diff * 5.0
            prev_foot_height = foot_height
            reward += float(reward_env)
        else:
            prev_distance = None  # 数据无效，重置
            prev_foot_height = None
            reward = float(reward_env)

        reward -= 0.02 * (abs(action_LegUpper) + abs(action_LegLower) + abs(action_Ankle))
        
        # 【新增】如果所有离散动作都为0，给予负奖励惩罚
        if discrete_upper == 0 and discrete_lower == 0 and discrete_ankle == 0:
            gate_activation["all_off"] += 1
            reward -= 1.0  # 惩罚值可调
            print(f"⚠️  警告：所有下半身离散动作都为0，给予惩罚 -1.0")
        
        # ========== return_all 计算 ==========
        # return_all: 当前episode的累积奖励总和
        # 每步执行后，将当前步的reward累加到return_all中
        return_all += reward
        steps += 1
        # 获取新的观察
        next_obs_img, next_obs_tensor = env.get_img(steps, obs_img)
        
        # 存储经验（除去明显无效样本）
        should_store = True
        if done == 1 and steps <= 2 and good != 1 and reward == 0:
            should_store = False
            print(f"  跳过无效样本：done={done}, steps={steps}, good={good}")

        if should_store:
            # 使用tai_agent存储转移经验
            tai_agent.store_transition(
                state=[obs_img, robot_state, robot_state],
                discrete_action=tai_discrete_action,
                continuous_action=tai_continuous_action,
                reward=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=tai_value,
                discrete_log_prob=tai_dict['discrete_log_prob'],
                continuous_log_prob=tai_continuous_log_prob
            )
            print(f"  已存储经验: reward={reward:.4f}")
                        
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps >= 20:
            done = 1
            
        # 定期保存模型
        if episode % 400 == 0 and done == 1:
            save_path = os.path.join(tai_checkpoint_dir, f"tai_agent_{total_episode}_{episode}.ckpt")
            print(f"保存模型到: {save_path}")
            checkpoint = {
                "episode": episode,
                "policy_tai": tai_agent.policy.state_dict(),
                "optimizer_tai": tai_agent.optimizer.state_dict()
            }
            torch.save(checkpoint, save_path)
        
        #学习过程
        if episode > 0 and done == 1:
            # 【方案A】使用训练管理器控制学习频率
            if training_manager is not None:
                training_manager.increment_tai()
                if training_manager.should_learn_tai():
                    # ========== Loss 计算 ==========
                    # 使用tai_agent进行学习，返回离散和连续两个损失
                    loss_discrete, loss_continuous = tai_agent.learn()
                    print("=" * 60)
                    print(f"【抬腿模型学习】{training_manager.get_status()}")
                    print(f"【第 {episode} 回合训练完成】")
                    print(f"  累积奖励 (return_all): {return_all:.4f}")
                    print(f"  目标达成 (goal): {goal}")
                    print(f"  离散损失 (loss_discrete): {loss_discrete:.6f}")
                    print(f"  连续损失 (loss_continuous): {loss_continuous:.6f}")
                    print(f"  总损失 (total loss): {loss_discrete + loss_continuous:.6f}")
                    if gate_activation["steps"] > 0:
                        print(f"【门控激活率】")
                        print(f"  LegUpper 激活率: {gate_activation['upper'] / gate_activation['steps']:.2%}")
                        print(f"  LegLower 激活率: {gate_activation['lower'] / gate_activation['steps']:.2%}")
                        print(f"  Ankle 激活率: {gate_activation['ankle'] / gate_activation['steps']:.2%}")
                        print(f"  全关闭次数: {gate_activation['all_off']} / {int(gate_activation['steps'])}")
                    print("=" * 60)
                    
                    # 记录损失值
                else:
                    # 累积经验但不学习
                    print(f"【抬腿模型累积经验】{training_manager.get_status()}")
                    loss_discrete, loss_continuous = 0, 0
            else:
                # 如果没有training_manager，使用原始逻辑（向后兼容）
                # ========== Loss 计算 ==========
                # 使用tai_agent进行学习，返回离散和连续两个损失
                loss_discrete, loss_continuous = tai_agent.learn()
                print("=" * 60)
                print(f"【第 {episode} 回合训练完成】")
                print(f"  累积奖励 (return_all): {return_all:.4f}")
                print(f"  目标达成 (goal): {goal}")
                print(f"  离散损失 (loss_discrete): {loss_discrete:.6f}")
                print(f"  连续损失 (loss_continuous): {loss_continuous:.6f}")
                print(f"  总损失 (total loss): {loss_discrete + loss_continuous:.6f}")
                if gate_activation["steps"] > 0:
                    print(f"【门控激活率】")
                    print(f"  LegUpper 激活率: {gate_activation['upper'] / gate_activation['steps']:.2%}")
                    print(f"  LegLower 激活率: {gate_activation['lower'] / gate_activation['steps']:.2%}")
                    print(f"  Ankle 激活率: {gate_activation['ankle'] / gate_activation['steps']:.2%}")
                    print(f"  全关闭次数: {gate_activation['all_off']} / {int(gate_activation['steps'])}")
                print("=" * 60)
            
            # 记录损失值
            total_loss = loss_discrete + loss_continuous
            log_writer_tai.add(loss=total_loss)
            log_writer_tai.add(loss_discrete=loss_discrete)
            log_writer_tai.add(loss_continuous=loss_continuous)
            # ========== 记录结果 ==========
            # return_all: 当前episode的累积奖励（所有步的reward之和）
            # goal: 环境返回的目标达成标志（在RobotRun2.py中计算）
            #   - goal=1: 脚部成功接触到梯子（touch传感器触发且距离较近）
            #   - goal=0: 未达到目标或距离太远
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)
            if gate_activation["steps"] > 0:
                log_writer_tai.add(
                    gate_upper_ratio=gate_activation["upper"] / gate_activation["steps"]
                )
                log_writer_tai.add(
                    gate_lower_ratio=gate_activation["lower"] / gate_activation["steps"]
                )
                log_writer_tai.add(
                    gate_ankle_ratio=gate_activation["ankle"] / gate_activation["steps"]
                )
                log_writer_tai.add(
                    gate_all_off_ratio=gate_activation["all_off"] / gate_activation["steps"]
                )
            gate_activation = {"upper": 0.0, "lower": 0.0, "ankle": 0.0, "all_off": 0, "steps": 0}
            
            # 如果回合结束，重置环境
        print("done:", done)


        if done == 1 or steps > 20:
            print("抬腿回合结束，重置环境...")
            env.darwin._set_left_leg_initpose()  # 重置左腿
            env.darwin.robot_reset()  # 重置环境
            env.reset()
            # 增加初始稳定时间
            print("等待稳定...")
            for _ in range(40):  # 增加40个时间步的稳定时间
                env.robot.step(env.timestep)
                
           
            print("等待一秒...")
            env.wait(1000)
            imgs = []
            steps = 0
            #episode += 1
            obs, obs_tensor = env.get_img(steps, imgs)
            robot_state = env.get_robot_state()
            
            log_writer_tai.save_tai(log_file_latest_tai)
            break
        