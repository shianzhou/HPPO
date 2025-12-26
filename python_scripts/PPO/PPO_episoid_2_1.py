import math
import torch
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1
from python_scripts.PPO.PPO_PPOnet_2 import PPO2 
from python_scripts.PPO.hppo import HPPO
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.PPO_Log_write import Log_write

def PPO_tai_episoid( existing_env=None ,total_episode=0, episode=0, log_writer_tai=None, log_file_latest_tai=None):

    hppo_agent_tai= hppo(num_servos=3, node_num=19, env_information=None)


    # 使用已有的环境实例或创建新的
    if existing_env is not None:
        env = existing_env
    else:
        env = Environment()

    print("开始抬腿！")
    # HPPO 3维离散开关：LegUpper, LegLower, Ankle
    hppo_switch_tai = HPPO(num_servos=3, node_num=19, env_information=None)
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
    
    # 记录上一次实际发送到环境的动作（用于离散=0时保持不变）
    prev_U = 0.0
    prev_L = 0.0
    prev_A = 0.0
    while True:
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        # 将机器人状态转换为PPO状态，与PPO_episoid_1.py保持一致
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]
        # 选择动作
        dict = hppo_agent_tai.choose_action(episode_num=episode,
                                        obs=[obs_img, robot_state],
                                        x_graph=robot_state)

        d_action = dict['discrete_action']

        action_LegUpper = dict['continuous_action'][0]
        action_LegLower = dict['continuous_action'][1]
        action_Ankle = dict['continuous_action'][2]
        log_prob_LegUpper = dict['continuous_log_prob'][0]
        log_prob_LegLower = dict['continuous_log_prob'][1]
        log_prob_Ankle = dict['continuous_log_prob'][2]
        value = dict['value']


        dU, dL, dA = float(d_action[0]), float(d_action[1]), float(d_action[2])

        cur_U = float(action_LegUpper)
        cur_L = float(action_LegLower)
        cur_A = float(action_Ankle)
        # 若离散为0，则保持上一时刻指令
        m_LegUpper = prev_U if int(dU) == 0 else cur_U
        m_LegLower = prev_L if int(dL) == 0 else cur_L
        m_Ankle = prev_A if int(dA) == 0 else cur_A
        
        # 分别添加动作、对数概率和状态价值到日志
        log_writer_tai.add_action_tai(action_LegUpper, action_LegLower, action_Ankle)
        log_writer_tai.add_log_prob_tai(log_prob_LegUpper, log_prob_LegLower, log_prob_Ankle)
        log_writer_tai.add_value_tai(value, value, value)

        print("第", steps + 1, "步")
        print(f"{float(action_LegUpper):.4f}×{int(dU)}, {float(action_LegLower):.4f}×{int(dL)}, {float(action_Ankle):.4f}×{int(dA)}")

        # 获取GPS数据
        gps_values = env.print_gps()
            
        # 设置抓取器状态
        catch_flag = 0.0
            
        # 生成图像名称
        img_name = f"img{steps}.png"
            
        # 执行动作
        next_state, reward, done, good, goal, count = env.step2(
            robot_state, m_LegUpper, m_LegLower, m_Ankle, steps, catch_flag, 
            gps_values[4], gps_values[0], gps_values[1], gps_values[2], gps_values[3],
        )
        # 更新上一时刻动作
        prev_U = m_LegUpper
        prev_L = m_LegLower
        prev_A = m_Ankle
        
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
            #rpm_2.append((robot_state, action, reward, next_state, done))
            # 将数据存储到PPO2对象内部
            # ppo2_LegUpper.store_transition_tai(
            #     state=[obs_img, robot_state, robot_state],
            #     actions=action_LegUpper,
            #     rewards=reward,
            #     next_state=[next_obs_img, robot_state, next_state],
            #     done=done,
            #     value=value_LegUpper,
            #     log_prob=log_prob_LegUpper
            # )
            # ppo2_LegLower.store_transition_tai(
            #     state=[obs_img, robot_state, robot_state],
            #     actions=action_LegLower,
            #     rewards=reward,
            #     next_state=[next_obs_img, robot_state, next_state],
            #     done=done,
            #     value=value_LegLower,
            #     log_prob=log_prob_LegLower
            # )
            # ppo2_Ankle.store_transition_tai(
            #     state=[obs_img, robot_state, robot_state],
            #     actions=action_Ankle,
            #     rewards=reward,
            #     next_state=[next_obs_img, robot_state, next_state],
            #     done=done,
            #     value=value_Ankle,
            #     log_prob=log_prob_Ankle
            # )
            # # 存储离散HPPO
            # hppo_switch_tai.store_transition(
            #     state=[obs_img, robot_state, robot_state],
            #     action=d_action,
            #     reward=reward,
            #     next_state=[next_obs_img, robot_state, next_state],
            #     done=done,
            #     value=d_value,
            #     log_prob=d_log_prob
            # )

            hppo_agent_tai.store_transition(
                state=[obs_img, robot_state, robot_state],
                discrete_action=d_action,
                continuous_action=dict['continuous_action'],
                reward=reward,
                next_state=[next_obs_img, next_state, next_state],
                done=done,
                value=value,
                discrete_log_prob=dict['discrete_log_prob'],
                continuous_log_prob=dict['continuous_log_prob']
            )
                        
        # 更新状态
        robot_state = env.get_robot_state()
        obs_tensor = next_obs_tensor
            
        # 检查是否结束当前回合
        if steps > 20:
            done = 1
            
        # 定期保存模型
        if episode % 200 == 0 and done == 1:
            save_path = path_list['model_path_tai_PPO'] + f"/ppo_model_tai_{total_episode}_{episode}.ckpt"
            print(f"保存模型到: {save_path}")

            checkpoint = {
                "episode": episode,  # 只写一次即可
                # 上腿
                "policy_tai": hppo_agent_tai.policy.state_dict(),
                "optimizer_c": hppo_agent_tai.optimizer_c.state_dict(),
                "optimizer_d": hppo_agent_tai.optimizer_d.state_dict(),
                "optimizer_v": hppo_agent_tai.optimizer_v.state_dict(),
            }
            torch.save(checkpoint, save_path)
        
        #学习过程
        if episode > 0 and done == 1:


            loss_d, loss_c = hppo_agent_tai.learn()

            loss1, loss2 = loss_d, loss_c
            print('loss_discrete:', loss1, 'loss_continuous:', loss2)
            
            # 分别记录三个智能体的loss值
            #log_writer_tai.add_loss_tai(loss_LegUpper, loss_LegLower, loss_Ankle, loss_hppo, loss)
            # need rewrite log mean

            log_writer_tai.add_loss_hppo_tai(loss1,loss2)
            # 记录结果
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)
            
            # 如果回合结束，重置环境
        print("done:", done)

        # #测试代码
        # if done == 1 :
        #     temp_loss = ppo2.learn()
        #     print("测试代码运行成功")

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
            
            log_writer_tai.save_tai(log_file_latest_tai)
            break
    