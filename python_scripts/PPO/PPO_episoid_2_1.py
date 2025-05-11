import math
import torch
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1
from python_scripts.PPO.PPO_PPOnet_2 import PPO2 

def PPO_tai_episoid(ppo2=None, existing_env=None ,total_episode=0, episode=0, rpm_2=None, log_writer_tai=None, log_file_latest_tai=None):

    if ppo2 is None:
        ppo2 = PPO2()
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
    imgs = []
    goal = 0
    done = 0
    reward = 0
    steps = 0
    catch_flag = 0
    # 获取观察和状态
    print("____________________")

    # 记录回合数
    log_writer_tai.add(episode_num=total_episode)

    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        # 将机器人状态转换为PPO状态，与PPO_episoid_1.py保持一致
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]
        # 选择动作
        action, log_prob, value = ppo2.choose_action(episode_num=episode, 
                                       obs=[obs_img, ppo_state],
                                       x_graph=robot_state)

        # 分别添加动作、对数概率和状态价值到日志
        log_writer_tai.add_action(action)
        log_writer_tai.add_log_prob(log_prob)
        log_writer_tai.add_value(value)

        print("第", steps + 1, "步")
        print(f"选择动作: {action}")

        # 获取GPS数据
        gps_values = env.print_gps()
            
        # 设置抓取器状态
        catch_flag = 0.0
            
        # 生成图像名称
        img_name = f"img{steps}.png"
            
        # 执行动作
        next_state, reward, done, good, goal, count = env.step2(
            robot_state, action, steps, catch_flag, 
            gps_values[4], gps_values[0], gps_values[1], gps_values[2], gps_values[3],
        )
        
        # 计算奖励
        if count == 1:
            x1 = gps_goal1[0] - gps_values[4][1]
            y1 = gps_goal1[1] - gps_values[4][2]
            distance = math.sqrt(x1 * x1 + y1 * y1)
                
            if distance > 0.06:
                reward1 = 0
            elif distance > 0.03:
                reward1 = 0.1
            else:
                reward1 = 1
                    
            reward = reward1
            
        return_all += reward
        steps += 1
        # 获取新的观察
        next_obs_img, next_obs_tensor = env.get_img(steps, obs_img)
        
        # 存储经验
        if good == 1:
            rpm_2.append((robot_state, action, reward, next_state, done))
            
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps > 20:
            done = 1
            
        # 定期保存模型
        if episode % 50 == 0:
            save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")
            checkpoint = {
                'policy': ppo2.policy.state_dict(),
                'optimizer': ppo2.optimizer.state_dict(),
                'episode': episode_num
            }
            torch.save(checkpoint, save_path)
        
        #学习过程
        if len(rpm_2) > 2000 and done == 1:
            # 如果达到目标，保存模型
            if goal == 1:
                save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
                checkpoint = {
                    'policy': ppo2.policy.state_dict(),
                    'optimizer': ppo2.optimizer.state_dict(),
                    'episode': episode_num
                }
                torch.save(checkpoint, save_path)
                
                # 学习
                loss = ppo2.learn(rpm_2)

                # 记录损失值
                log_writer_tai.add(loss=loss)
                
            # 记录结果
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)
            
            # 如果回合结束，重置环境
        print("done:", done)
        if done == 1 or steps > 20:
            print("抬腿回合结束，重置环境...")
            env.darwin._set_left_leg_initpose()  # 重置环境
            print("等待一秒...")
            env.wait(1000)
            imgs = []
            steps = 0
            #episode += 1
            obs, obs_tensor = env.get_img(steps, imgs)
            robot_state = env.get_robot_state()
            log_writer_tai.clear()
            log_writer_tai.save_tai(log_file_latest_tai)
            break