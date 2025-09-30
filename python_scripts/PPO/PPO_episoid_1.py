# 测试
import torch
from python_scripts.PPO.PPO_PPOnet_2 import PPO2
from python_scripts.PPO.PPO_PPOnet import PPO
from python_scripts.PPO.hppo import HPPO
from python_scripts.PPO.Replay_memory import ReplayMemory
from python_scripts.PPO.Replay_memory_2 import ReplayMemory_2
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.PPO_Log_write import Log_write

def PPO_episoid_1(model_path=None, max_steps_per_episode=500):   
    ppo_arm = PPO(node_num=19, env_information=None)  # 创建PPO对象
    ppo_shoulder = PPO(node_num=19, env_information=None)  # 创建PPO对象

    # 混合动作：两个离散开关（肩、臂是否运动）
    hppo_switch_catch = HPPO(num_servos=2, node_num=19, env_information=None)

    ppo2_LegUpper = PPO2(node_num=19, env_information=None)  # 创建PPO2对象
    ppo2_LegLower = PPO2(node_num=19, env_information=None)  # 创建PPO2对象
    ppo2_Ankle = PPO2(node_num=19, env_information=None)  # 创建PPO2对象

    # 初始化日志写入器
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()  # 创建抬腿日志写入器

    tai_episoid = 1
    import os
    import glob
    import re
    # 查找现有的日志文件，确定最新的编号
    # 抓取阶段：
    log_pattern = os.path.join(path_list['catch_log_path_PPO'], 'catch_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'catch_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_catch = os.path.join(path_list['catch_log_path_PPO'], f"catch_log_{new_log_num}.json")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")

    # 抬腿阶段：
    log_pattern = os.path.join(path_list['tai_log_path_PPO'], 'tai_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'tai_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_tai = os.path.join(path_list['tai_log_path_PPO'], f"tai_log_{new_log_num}.json")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")

    # 加载模型
    # 抓取模型加载
    if model_path:  # 如果指定了模型路径
        try:
            # 从指定路径加载模型
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                # 如果是保存的字典格式 {'policy': state_dict, ...}
                ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'])
                ppo_arm.policy.load_state_dict(checkpoint['policy_arm'])
                # 如果需要加载优化器状态
                if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                    ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                    ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                episode_start = int(model_path.split('_')[-1].split('.')[0])
                print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            else:
                # 如果是直接保存的模型或状态字典
                ppo_shoulder.policy.load_state_dict(checkpoint)
                ppo_arm.policy.load_state_dict(checkpoint)
                print("从指定模型加载: {model_path}，模型加载成功！(旧格式)")
                episode_start = 0
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:  # 如果没有指定模型路径，使用原来的自动查找逻辑
        # 获取所有模型文件
        model_files = glob.glob(path_list['model_path_catch_PPO'] + '/ppo_model_*.ckpt')
        if model_files:
            # 按文件名中的数字排序，获取最新的模型文件
            latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            episode_start = int(latest_model.split('_')[-1].split('.')[0])
            print(f"找到最新抓取模型: {latest_model}，从周期 {episode_start} 继续训练")
            
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                    # 如果是保存的字典格式 {'policy': state_dict, ...}
                    ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'])
                    ppo_arm.policy.load_state_dict(checkpoint['policy_arm'])
                    # 如果需要加载优化器状态
                    if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                        ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                    if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                        ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                    print("抓取模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo_shoulder.policy.load_state_dict(checkpoint)
                    ppo_arm.policy.load_state_dict(checkpoint)
                    print("抓取模型加载成功！(旧格式)")
            except Exception as e:
                print(f"抓取模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的抓取模型，从头开始训练")
            episode_start = 0
    
    # 抬腿模型加载
    model_files_tai = glob.glob(path_list['model_path_tai_PPO'] + '/ppo_model_tai_*.ckpt')
    if model_files_tai:
        try:
            # 按新的文件名格式排序：ppo_model_tai_{total_episoid}_{episode}.ckpt
            # 定义一个函数来提取total_episoid和episode
            def extract_numbers(filename):
                # 从文件名中提取数字部分
                parts = filename.split('_')
                if len(parts) >= 5:  # 确保文件名格式正确
                    try:
                        total_ep = int(parts[-2])  # 倒数第二个是total_episoid
                        ep = int(parts[-1].split('.')[0])  # 最后一个是episode（去掉.ckpt）
                        return (total_ep, ep)
                    except (ValueError, IndexError):
                        return (0, 0)  # 解析失败时返回默认值
                return (0, 0)
            
            # 按照total_episoid和episode排序，找出最新的模型
            latest_model = max(model_files_tai, key=extract_numbers)
            total_ep, ep = extract_numbers(latest_model)
            print(f"找到最新抬腿模型: {latest_model}，总周期: {total_ep}，抬腿周期: {ep}")
            tai_episoid = ep
            print(f"抬腿模型从周期 {tai_episoid} 继续训练")
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                # 判断是否为新版字典格式（包含各关节 policy 键）
                if isinstance(checkpoint, dict) and 'policy_LegUpper' in checkpoint:
                    # 如果是保存的字典格式 {'policy_LegUpper': state_dict, ...}
                    ppo2_LegUpper.policy.load_state_dict(checkpoint['policy_LegUpper'])
                    ppo2_LegLower.policy.load_state_dict(checkpoint['policy_LegLower'])
                    ppo2_Ankle.policy.load_state_dict(checkpoint['policy_Ankle'])
                    # 如果需要加载优化器状态
                    if 'optimizer_LegUpper' in checkpoint and ppo2_LegUpper.optimizer:
                        ppo2_LegUpper.optimizer.load_state_dict(checkpoint['optimizer_LegUpper'])
                    if 'optimizer_LegLower' in checkpoint and ppo2_LegLower.optimizer:
                        ppo2_LegLower.optimizer.load_state_dict(checkpoint['optimizer_LegLower'])
                    if 'optimizer_Ankle' in checkpoint and ppo2_Ankle.optimizer:
                        ppo2_Ankle.optimizer.load_state_dict(checkpoint['optimizer_Ankle'])
                    print("抬腿模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo2_LegUpper.policy.load_state_dict(checkpoint)
                    ppo2_LegLower.policy.load_state_dict(checkpoint)
                    ppo2_Ankle.policy.load_state_dict(checkpoint)
                    print("抬腿模型加载成功！(旧格式)")
            except Exception as e:
                print(f"抬腿模型加载失败: {e}")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
    else:
        print("未找到已保存的抬腿模型，从头开始训练")




    episode_num = episode_start  # 初始化回合计数器
    #rpm = ReplayMemory(100000)  # 创建经验回放缓存
    #rpm_2 = ReplayMemory_2(100000)
    env = Environment()
    success_catch = 0                  # 抓取成功次数

    for i in range(episode_start, episode_start + 1000):  # 从episode_start开始，最多再训练10000个周期
        log_writer_catch.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期") # 打印当前周期
        env.reset()
        env.wait(500)   # 等待500ms
        imgs = []  # 初始化图像列表
        steps = 0  # 初始化步数
        return_all = 0  # 初始化总奖励
        obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
        # log_writer.add(obs_img=obs_img, steps=steps)
        robot_state = env.get_robot_state()  # 获取机器人状态
        # print(f'robot_state: {robot_state}')
        # print(f'robot_state_len: {len(robot_state)}')
        print("____________________")  # 打印初始状态
        # 记录上一次实际发送到环境的动作（用于离散=0时保持不变）
        prev_shoulder_action = 0.0
        prev_arm_action = 0.0
        while True:
            # print(f'第{episode_num}周期，第{steps}步')
            ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]  # 将机器人状态转换为ppo状态
            # log_writer.add(ppo_state=ppo_state, steps=steps)
            obs = (obs_tensor, robot_state)
            # log_writer.add(obs=obs, steps=steps)
            # 将机器人状态转换为张量
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).to(device)
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(1).to(device)  # 添加维度
            # 输入次数、状态，选择动作
            action_shoulder , log_prob_shoulder , value_shoulder = ppo_shoulder.choose_action(episode_num=i, 
                                  obs=obs,
                                  x_graph=robot_state,
                                  action_type='shoulder')
            action_arm , log_prob_arm , value_arm = ppo_arm.choose_action(episode_num=i, 
                                  obs=obs,
                                  x_graph=robot_state,
                                  action_type='arm')
            # 离散开关：d0->shoulder, d1->arm (0/1)
            d_action, d_log_prob, d_value = hppo_switch_catch.choose_action(
                episode_num=i,
                obs=obs,
                x_graph=robot_state
            )
            # 转为0/1掩码
            d0 = float(d_action[0])
            d1 = float(d_action[1])
            cur_shoulder = float(action_shoulder.item())
            cur_arm = float(action_arm.item())
            # 若离散为0，则保持上一时刻指令
            masked_shoulder = prev_shoulder_action if int(d0) == 0 else cur_shoulder
            masked_arm = prev_arm_action if int(d1) == 0 else cur_arm
            print(f'第{i}周期，第{steps}步, 离散数值：({int(d0)},{int(d1)}), 连续: ({action_shoulder.item():.4f},{action_arm.item():.4f}), 实际: ({masked_shoulder:.4f},{masked_arm:.4f})')
            
            # # 简化动作处理逻辑
            # if isinstance(a, tuple):
            #     # 如果是元组，取第一个元素作为动作
            #     action_value = a[0]
            # else:
            #     # 否则直接使用a
            #     action_value = a
            # # 确保action_value是标量
            # if hasattr(action_value, 'cpu'):
            #     action_value = action_value.cpu()
            # if hasattr(action_value, 'item'):
            #     action_value = action_value.item()
                
            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
            if steps >= 19:  # 如果步数大于等于19
                catch_flag = 1.0  # 抓取器状态为1.0
            else:
                catch_flag = 0.0  # 抓取器状态为0.0
            img_name = "img" + str(steps) + ".png"  # 图像名称
            # print("action:", a)
            # 分别添加动作、对数概率和状态价值到日志
            log_writer_catch.add_action_catch(action_shoulder, action_arm) 
            log_writer_catch.add_log_prob_catch(log_prob_shoulder, log_prob_arm)
            log_writer_catch.add_value_catch(value_shoulder, value_arm)
            # 执行一步动作
            next_state, reward, done, good, goal, count = env.step(
                robot_state,
                masked_shoulder,
                masked_arm,
                steps,
                catch_flag,
                gps1,
                gps2,
                gps3,
                gps4,
                img_name
            )
            # 更新上一时刻动作
            prev_shoulder_action = masked_shoulder
            prev_arm_action = masked_arm
            print(f'catch_flag: {catch_flag}')
            print(f'done: {done}')
            
            if count == 1:  # 如果计数器为1 
                gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
                x1 = gps_goal[0] - gps1[1]  # 计算目标位置与当前位置的差值
                y1 = gps_goal[1] - gps1[2]
                if x1 > -0.03 and y1 < 0.03:
                    reward1 = 1  # 奖励为1
                elif -0.05 < x1 < -0.03 and 0.03 < y1 < 0.05:
                    reward1 = 1  # 奖励为1
                else:
                    reward1 = 0  # 奖励为0
                reward = reward1  # 奖励为reward1
            return_all = return_all + reward  # 总奖励为当前奖励加上之前的总奖励
            steps += 1  # 步数加1
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)  # 获取下一个图像和图像张量
            next_obs = [next_obs_img, next_state]
            # print('获取下一个状态更新完毕')
            # 可以修改reward值让其训练速度加快
            if good == 1:  # 如果good为1
                # 将当前状态、动作、奖励、下一个状态、是否完成、是否达到目标添加到经验回放缓存中
                #rpm.append((obs_img, robot_state, action, log_prob, reward, done, value))
                # 同时将数据存储到PPO对象内部
                ppo_shoulder.store_transition_catch(
                    state=[obs_img, robot_state, robot_state],  # 包含图像、机器人状态和图神经网络输入
                    action_shoulder=action_shoulder,
                    action_arm=action_arm,                 
                    reward=reward,
                    next_state=[next_obs_img, next_state, next_state],  # 包含下一个图像、下一个状态和图神经网络输入
                    done=done,
                    value_shoulder=value_shoulder,
                    value_arm=value_arm,
                    log_prob_shoulder=log_prob_shoulder,
                    log_prob_arm=log_prob_arm
                )
                ppo_arm.store_transition_catch(
                    state=[obs_img, robot_state, robot_state],
                    action_shoulder=action_shoulder,
                    action_arm=action_arm,
                    reward=reward,
                    next_state=[next_obs_img, next_state, next_state],
                    done=done,
                    value_shoulder=value_shoulder,
                    value_arm=value_arm,
                    log_prob_shoulder=log_prob_shoulder,
                    log_prob_arm=log_prob_arm
                )
                # 存储离散HPPO轨迹
                hppo_switch_catch.store_transition(
                    state=[obs_img, robot_state, robot_state],
                    action=d_action,
                    reward=reward,
                    next_state=[next_obs_img, next_state, next_state],
                    done=done,
                    value=d_value,
                    log_prob=d_log_prob
                )
            robot_state = env.get_robot_state()  # 获取机器人状态



            obs_tensor = next_obs_tensor  # 更新图像张量
            #if temp < 5000:  # 如果经验回放缓存小于3000
                #episode_num = 0  # 计数器为0
            if i >= 0 and done == 1:  # 只有在buffer中存满了数据才会学习
                if goal == 1:  # 如果达到目标
                    print("goal = 1")
                    save_path = path_list['model_path_catch_PPO'] + '/ppo_model_%s.ckpt' % i  # 保存模型
                    checkpoint = {
                        'policy_shoulder': ppo_shoulder.policy.state_dict(),
                        'optimizer_shoulder': ppo_shoulder.optimizer.state_dict(),
                        'policy_arm': ppo_arm.policy.state_dict(),
                        'optimizer_arm': ppo_arm.optimizer.state_dict(),
                        'episode': i
                    }
                    torch.save(checkpoint, save_path)
                print("11111111111111111111111111111111111111111-303")
                loss_shoulder = ppo_shoulder.learn(action_type='shoulder')
                print("22222222222222222222222222222222222222222-305")
                loss_arm = ppo_arm.learn(action_type='arm')
                # 学习离散HPPO
                loss_hppo = hppo_switch_catch.learn()

                loss = loss_shoulder + loss_arm + loss_hppo

                print('loss_arm:', loss_arm)
                print('loss_shoulder:', loss_shoulder)
                print('loss_hppo:', loss_hppo)
                print('loss:', loss)
                
                # 分别记录三个智能体的loss值
                log_writer_catch.add_loss_catch(loss_shoulder, loss_arm, loss_hppo, loss)
                # 立即落盘，避免仅在回合结束保存导致当轮loss缺失
                try:
                    log_writer_catch.save_catch(log_file_latest_catch)
                except Exception as _e:
                    print(f"保存抓取loss到日志失败: {_e}")
               
                if i % 500 == 0 and i != 0:  # 每100步保存一次模型
                    save_path = path_list['model_path_catch_PPO'] + '/ppo_model_%s.ckpt' % i  # 保存模型
                    checkpoint = {
                        'policy_shoulder': ppo_shoulder.policy.state_dict(),
                        'optimizer_shoulder': ppo_shoulder.optimizer.state_dict(),
                        'policy_arm': ppo_arm.policy.state_dict(),
                        'optimizer_arm': ppo_arm.optimizer.state_dict(),
                        'episode': i
                    }
                    torch.save(checkpoint, save_path)

                log_writer_catch.add(return_all=return_all)
                # 写入目标
                log_writer_catch.add(goal=goal)
                
            success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

            if catch_flag == 1.0 or done == 1:  # 如果抓取器状态为1.0或完成
                # 写入重置标志
                # if(success_flag1 == 0):
                #     env.reset()  # 重置环境
                env.wait(100)  # 等待100ms
                imgs = []  # 初始化图像列表
                steps = 0  # 初始化步数
                episode_num = episode_num + 1  # 计数器加1
                # obs, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
                # robot_state = env.get_robot_state()  # 获取机器人状态
                log_writer_catch.clear()
                log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
                break

        if success_flag1 == 1:
            success_catch += 1
            log_writer_catch.add(success_catch=success_catch)
            print("success_catch:", success_catch)
            print("抓取成功，开始抬腿训练...")
            total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(ppo2_LegUpper=ppo2_LegUpper, ppo2_LegLower=ppo2_LegLower, ppo2_Ankle=ppo2_Ankle, existing_env=env, total_episode=total_episode, episode=tai_episoid, log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai)
            tai_episoid += 1 

    
    log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env