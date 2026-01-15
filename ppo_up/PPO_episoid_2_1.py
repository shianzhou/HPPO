import math
import numpy as np
import torch
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1, Darwin_config
from python_scripts.PPO.PPO_PPOnet_2 import PPO2 
from python_scripts.utils.sensor_utils import wait_for_sensors_stable, reset_environment 
from python_scripts.PPO_Log_write import Log_write
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
def PPO_tai_episoid(ppo2_LegUpper=None, ppo2_LegLower=None, ppo2_Ankle=None, existing_env=None ,total_episode=0, episode=0, log_writer_tai=None, log_file_latest_tai=None, hppo_agent=None):

    if ppo2_LegUpper is None:
        ppo2_LegUpper = PPO2()
    if ppo2_LegLower is None:
        ppo2_LegLower = PPO2()
    if ppo2_Ankle is None:
        ppo2_Ankle = PPO2()
    # 使用已有的环境实例或创建新的
    
    if existing_env is None:
        raise RuntimeError(
            "PPO_tai_episoid 必须传入 existing_env（复用同一个 Environment/Robot），"
            "禁止在 PPO_episoid_2_1.py 内部 Environment() 新建。"
        )
    env = existing_env

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
    last_action_LegUpper = 0.0  # LAnklePitch
    last_action_LegLower = 0.0 # LAnkleRoll
    last_action_Ankle = 0.0    # LAnkleRoll

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
        # 选择动作
        action_LegUpper, log_prob_LegUpper, value_LegUpper = ppo2_LegUpper.choose_action(episode_num=episode, 
                                       obs=[obs_img, robot_state],
                                       x_graph=robot_state)
        action_LegLower, log_prob_LegLower, value_LegLower = ppo2_LegLower.choose_action(episode_num=episode, 
                                       obs=[obs_img, robot_state],
                                       x_graph=robot_state)
        action_Ankle, log_prob_Ankle, value_Ankle = ppo2_Ankle.choose_action(episode_num=episode, 
                                       obs=[obs_img, robot_state],
                                       x_graph=robot_state)
        if hppo_agent is not None:
            lower_gate, hppo_log_prob_lower, hppo_value_lower = hppo_agent.choose_action(
                group='lower',
                obs=(obs_tensor, robot_state),
                x_graph=robot_state,
                episode_num=episode
            )
        else:
            lower_gate = np.ones(3)
            hppo_log_prob_lower = 0.0
            hppo_value_lower = 0.0
        gate_upper = float(lower_gate[0])
        gate_lower = float(lower_gate[1])
        gate_ankle = float(lower_gate[2])

        # 保存原始动作值（用于调试）
        action_LegUpper_original = action_LegUpper
        action_LegLower_original = action_LegLower
        action_Ankle_original = action_Ankle

        # 根据门控值决定是使用新动作还是保持上一动作
        if gate_upper < 0.5:
            action_LegUpper_exec = last_action_LegUpper
        else:
            action_LegUpper_exec = action_LegUpper

        if gate_lower < 0.5:
            action_LegLower_exec = last_action_LegLower
        else:
            action_LegLower_exec = action_LegLower

        if gate_ankle < 0.5:
            action_Ankle_exec = last_action_Ankle
        else:
            action_Ankle_exec = action_Ankle

        # 更新 "last_action" 变量以备下一步使用
        last_action_LegUpper = action_LegUpper_exec
        last_action_LegLower = action_LegLower_exec
        last_action_Ankle = action_Ankle_exec

        log_prob_LegUpper = ppo2_LegUpper.evaluate_log_prob(
            obs=[obs_img, robot_state], x_graph=robot_state, action=float(action_LegUpper_exec)
        )
        log_prob_LegLower = ppo2_LegLower.evaluate_log_prob(
            obs=[obs_img, robot_state], x_graph=robot_state, action=float(action_LegLower_exec)
        )
        log_prob_Ankle = ppo2_Ankle.evaluate_log_prob(
            obs=[obs_img, robot_state], x_graph=robot_state, action=float(action_Ankle_exec)
        )
        gate_activation["steps"] += 1
        gate_activation["upper"] += gate_upper
        gate_activation["lower"] += gate_lower
        gate_activation["ankle"] += gate_ankle
        
        # 分别添加动作、对数概率和状态价值到日志
        log_writer_tai.add_action_tai(action_LegUpper_exec, action_LegLower_exec, action_Ankle_exec)
        log_writer_tai.add_log_prob_tai(log_prob_LegUpper, log_prob_LegLower, log_prob_Ankle)
        log_writer_tai.add_value_tai(value_LegUpper, value_LegLower, value_Ankle)

        print("第", steps + 1, "步")
        print(f"【HPPO门控控制】离散动作: [{gate_upper:.0f}, {gate_lower:.0f}, {gate_ankle:.0f}]")
        print(f"【原始动作】LegUpper: {action_LegUpper_original:.4f}, LegLower: {action_LegLower_original:.4f}, Ankle: {action_Ankle_original:.4f}")
        print(f"【最终动作】LegUpper: {action_LegUpper_exec:.4f}, LegLower: {action_LegLower_exec:.4f}, Ankle: {action_Ankle_exec:.4f}")

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


        reward = 0.0
        foot_x = float(gps_values[4][1])      # 脚高度（X）
        foot_y = float(gps_values[4][2])      # 脚前后位置（y）
        
        goal_x = 0.058  # 目标高度
        goal_y = 0.023  # 目标水平位置
        
        # ===== 改进1：加权距离计算 =====
        # 给高度更高的权重（对台阶任务更关键）
        height_weight = 2.0  # 高度权重因子
        horizontal_weight = 1.0
        
        distance_x = abs(goal_x - foot_x)
        distance_y = abs(goal_y - foot_y)
        
        # 加权总距离
        weighted_distance = math.sqrt(
            (height_weight * distance_x)**2 + 
            (horizontal_weight * distance_y)**2
        )
        
        # ===== 改进2：平滑的距离奖励曲线 =====
        # 使用指数衰减的奖励，避免奖励阶跃
        distance_reward = 0.0
        if weighted_distance < 0.1:  # 10cm内
            # 使用指数衰减：离目标越近奖励越高，但曲线更平滑
            distance_reward = math.exp(-10 * weighted_distance) * 30
        else:
            # 远距离时使用线性惩罚，但不要太重
            distance_reward = -weighted_distance * 1.5
        
        reward += distance_reward
        
        # ===== 改进3：分阶段奖励策略 =====
        if weighted_distance < 0.05:  # 5cm内进入精细调整阶段
            # 高度精度奖励（更重要的目标）
            if distance_x < 0.01:  # 高度误差小于1cm
                reward += 20 * (0.01 - distance_x)
            
            # 水平位置奖励
            if distance_y < 0.02:  # 水平误差小于2cm
                reward += 10 * (0.02 - distance_y)
                
            # 成功奖励（更严格的标准）
            if weighted_distance < 0.025 and goal == 1:
                reward += 80.0  # 更高的成功奖励
                print("🎉 精确踩到台阶！")
                
        # ===== 改进4：智能下踩奖励 =====
        if prev_foot_height is not None:
            height_change = foot_x - prev_foot_height
            
            # 只在接近目标时奖励下踩
            if weighted_distance < 0.06:
                if height_change < 0:  # 下踩
                    # 奖励与距离成反比：越近下踩奖励越高
                    proximity_factor = max(0, 1 - weighted_distance/0.06)
                    reward += abs(height_change) * 80 * proximity_factor
                else:  # 抬起
                    # 适度奖励抬起，避免脚卡住
                    reward += height_change * 3
            else:
                # 远离目标时，适度奖励调整高度
                if abs(height_change) > 0:
                    reward += height_change * 2
        
        # ===== 改进5：动态动作惩罚 =====
        # 惩罚与距离相关：越接近目标，动作惩罚越轻
        if weighted_distance < 0.04:
            action_penalty = 0.005  # 接近目标时轻度惩罚
        else:
            action_penalty = 0.01   # 远离目标时正常惩罚
        
        reward -= action_penalty * (
            abs(action_LegUpper_exec) +
            abs(action_LegLower_exec) +
            abs(action_Ankle_exec)
        )
        
        # ===== 门控惩罚保持 =====
        if gate_upper < 0.5 and gate_lower < 0.5 and gate_ankle < 0.5:
            reward -= 0.3
        
        # ===== 更新历史状态 =====
        prev_foot_height = foot_x
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
            ppo2_LegUpper.store_transition_tai(
                state=[obs_img, robot_state, robot_state],
                actions=float(action_LegUpper_exec),
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_LegUpper,
                log_prob=log_prob_LegUpper
            )

            ppo2_LegLower.store_transition_tai(
                state=[obs_img, robot_state, robot_state],
                actions=float(action_LegLower_exec),
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_LegLower,
                log_prob=log_prob_LegLower
            )

            ppo2_Ankle.store_transition_tai(
                state=[obs_img, robot_state, robot_state],
                actions=float(action_Ankle_exec),
                rewards=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                value=value_Ankle,
                log_prob=log_prob_Ankle
            )

            if hppo_agent is not None:
                hppo_agent.store_transition(
                    group='lower',
                    state=[obs_tensor, robot_state, robot_state],
                    action=lower_gate,
                    reward=reward,
                    next_state=[next_obs_tensor, robot_state, next_state],
                    done=done,
                    value=hppo_value_lower,
                    log_prob=hppo_log_prob_lower
                )
                        
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps >= 20:
            done = 1
            
        # 定期保存模型
        if episode % 400 == 0 and done == 1:
            save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")
            checkpoint = {
                "episode": episode,                      # 只写一次即可
                # 上腿
                "policy_LegUpper":    ppo2_LegUpper.policy.state_dict(),
                "optimizer_LegUpper": ppo2_LegUpper.optimizer.state_dict(),
                # 下腿
                "policy_LegLower":    ppo2_LegLower.policy.state_dict(),
                "optimizer_LegLower": ppo2_LegLower.optimizer.state_dict(),
                # 踝关节
                "policy_Ankle":       ppo2_Ankle.policy.state_dict(),
                "optimizer_Ankle":    ppo2_Ankle.optimizer.state_dict(),
                # HPPO智能体
                "policy_hppo_lower": hppo_agent.policy.state_dict(),
                "optimizer_hppo_lower": hppo_agent.optimizer.state_dict()
            }
            torch.save(checkpoint, save_path)
        
        #学习过程
        if episode > 0 and done == 1:
            # 如果达到目标，保存模型
            # save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            # checkpoint = {
            #         'policy': ppo2.policy.state_dict(),
            #         'optimizer': ppo2.optimizer.state_dict(),
            #         'episode': episode
            #     }
            # torch.save(checkpoint, save_path)
            # if goal == 1:
            #     save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            #     checkpoint = {
            #         'policy': ppo2.policy.state_dict(),
            #         'optimizer': ppo2.optimizer.state_dict(),
            #         'episode': episode
            #     }
            #     torch.save(checkpoint, save_path)
                
            # ========== Loss 计算 ==========
            # Loss计算流程（在PPO_PPOnet_2.py的learn方法中）：
            # 1. 计算GAE优势函数：advantages = GAE(rewards, values, dones)
            # 2. 计算回报：returns = advantages + values
            # 3. 策略损失：policy_loss = -min(ratio * advantages, clip(ratio) * advantages)
            #    - ratio = exp(new_log_prob - old_log_prob)
            #    - clip(ratio) = clamp(ratio, 1-clip_ratio, 1+clip_ratio)
            # 4. 值函数损失：value_loss = MSE(predicted_values, returns)
            # 5. 熵正则化：entropy = dist.entropy().mean()
            # 6. 总损失：loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            # 7. 三个舵机的loss分别计算后相加：loss = loss_LegUpper + loss_LegLower + loss_Ankle
            
            loss_LegUpper = ppo2_LegUpper.learn()  # LegUpper舵机的PPO损失
            print("loss_LegUpper:", loss_LegUpper)
            loss_LegLower = ppo2_LegLower.learn()  # LegLower舵机的PPO损失
            print("loss_LegLower:", loss_LegLower)
            loss_hppo_lower = 0.0
            if hppo_agent is not None:
                loss_hppo_lower = hppo_agent.learn('lower')  # HPPO智能体的损失（单独计算，不加入总损失）
                print("loss_hppo_lower:", loss_hppo_lower)
            loss_Ankle = ppo2_Ankle.learn()  # Ankle舵机的PPO损失
            print("loss_Ankle:", loss_Ankle)
            loss = loss_LegUpper + loss_LegLower + loss_Ankle + loss_hppo_lower  # 三个舵机的总损失
            # 记录损失值
            log_writer_tai.add(loss=loss)
            log_writer_tai.add(loss_LegUpper=loss_LegUpper)
            log_writer_tai.add(loss_LegLower=loss_LegLower)
            log_writer_tai.add(loss_Ankle=loss_Ankle)
            log_writer_tai.add(loss_hppo_lower=loss_hppo_lower)
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
        