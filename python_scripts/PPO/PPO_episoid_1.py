# 测试
import torch
from python_scripts.PPO.PPO_PPOnet_2 import PPO2
from python_scripts.PPO.PPO_PPOnet import PPO
from python_scripts.PPO.hppo import HPPO
from python_scripts.PPO.Replay_memory import ReplayMemory
from python_scripts.PPO.Replay_memory_2 import ReplayMemory_2
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.PPO.hppo import HPPO as d_hppo
# from Data_fusion import data_fusion
from typing import Optional
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.PPO_Log_write import Log_write


# ===== 路径与文件工具函数（统一管理） =====
import os
import glob
import re

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _next_log_file(dir_path: str, prefix: str) -> str:
    pattern = os.path.join(dir_path, f"{prefix}_*.json")
    existing = glob.glob(pattern)
    max_n = 0
    for p in existing:
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.json$", os.path.basename(p))
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except Exception:
                continue
    return os.path.join(dir_path, f"{prefix}_{max_n + 1}.json")

def _latest_catch_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None, 0
    def _num(f: str) -> int:
        m = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    selected = max(files, key=_num)
    return selected, _num(selected)

def _latest_tai_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "tai_agent_*_*.ckpt"))
    if not files:
        return None, 0, 0
    def _nums(f: str):
        b = os.path.basename(f).replace('.ckpt','')
        parts = b.split('_')
        try:
            return int(parts[-2]), int(parts[-1])
        except Exception:
            return (0, 0)
    selected = max(files, key=_nums)
    total, ep = _nums(selected)
    return selected, total, ep

def _latest_decision_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return None, 0
    def _num(f: str) -> int:
        m = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    selected = max(files, key=_num)
    return selected, _num(selected)

# ===== 模型加载工具函数（提炼提高可读性） =====
def load_catch_model(model_path: str, hppo_agent, catch_dir: str) -> int:
    """加载抓取模型，优先使用指定路径；否则从目录中选择最新。返回 episode_start。"""
    episode_start = 0
    if model_path:
        try:
            ckpt = torch.load(model_path)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                try:
                    episode_start = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
                    print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
                except Exception:
                    pass
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print(f"从指定模型加载: {model_path}，模型加载成功！(旧格式)")
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
        return episode_start

    # 未指定路径，查找目录最新
    selected_model, episode_start = _latest_catch_ckpt(catch_dir)
    if selected_model:
        try:
            ckpt = torch.load(selected_model)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print("抓取模型加载成功！")
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print("抓取模型加载成功！(旧格式)")
        except Exception as e:
            print(f"抓取模型加载失败: {e}")
            episode_start = 0
    else:
        print("未找到已保存的抓取模型，从头开始训练")
        episode_start = 0
    return episode_start

def load_tai_model(tai_agent, tai_dir: str, default_episode: int = 1) -> int:
    """加载抬腿模型，仅从新目录选择最新。返回抬腿起始回合。"""
    selected_tai, _, ep = _latest_tai_ckpt(tai_dir)
    if selected_tai:
        print(f"找到最新抬腿模型: {selected_tai}，抬腿周期: {ep}")
        try:
            ckpt = torch.load(selected_tai)
            if isinstance(ckpt, dict) and 'policy_tai' in ckpt:
                tai_agent.policy.load_state_dict(ckpt['policy_tai'])
                if 'optimizer_tai' in ckpt and tai_agent.optimizer:
                    tai_agent.optimizer.load_state_dict(ckpt['optimizer_tai'])
            print("抬腿模型加载成功！")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
        return ep
    else:
        print("未找到已保存的抬腿模型，从头开始训练")
        return default_episode

def load_decision_model(decision_agent, dec_dir: Optional[str]) -> int:
    """可选加载决策模型（若目录存在且有文件）。返回起始决策回合编号。"""
    if not dec_dir:
        return 0
    latest_dec, dec_ep = _latest_decision_ckpt(dec_dir)
    if latest_dec:
        try:
            ckpt = torch.load(latest_dec)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                decision_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer' in ckpt and decision_agent.optimizer:
                    decision_agent.optimizer.load_state_dict(ckpt['optimizer'])
                print(f"决策模型加载成功: {latest_dec}")
            return dec_ep
        except Exception as e:
            print(f"决策模型加载失败: {e}")
    return 0


def PPO_episoid_1(model_path=None, max_steps_per_episode=5):
    # ===== 智能体实例化（统一放置） =====
    hppo_agent = hppo(num_servos=2, node_num=19, env_information=None)          # 抓取阶段智能体
    tai_agent = hppo(num_servos=3, node_num=19, env_information=None)       # 抬腿阶段智能体（复用）
    decision_hppo_agent = d_hppo(num_servos=1, node_num=19, env_information=None)  # 上层决策智能体

    # ===== 日志写入器 =====
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()    # 创建抬腿日志写入器
    log_writer_decision = Log_write()  # 创建决策日志写入器

    # ===== 基础计数 =====
    tai_episoid = 1

    # ===== 模型保存目录（统一，使用配置的新路径） =====
    catch_checkpoint_dir = path_list['model_path_catch_PPO_h']
    decision_checkpoint_dir = path_list['model_path_decision_PPO_h']
    _ensure_dir(catch_checkpoint_dir)
    _ensure_dir(decision_checkpoint_dir)

    # ===== 日志文件（自动递增编号） =====
    # 确保日志目录存在
    _ensure_dir(path_list['catch_log_path_PPO'])
    _ensure_dir(path_list['tai_log_path_PPO'])
    _ensure_dir(path_list['decision_log_path_PPO'])
    
    log_file_latest_catch = _next_log_file(path_list['catch_log_path_PPO'], 'catch_log')
    log_file_latest_tai = _next_log_file(path_list['tai_log_path_PPO'], 'tai_log')
    log_file_latest_decision = _next_log_file(path_list['decision_log_path_PPO'], 'decision_log')
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")
    print(f"将使用新决策的日志目录: {log_file_latest_decision}")

    # ===== 模型加载（函数化） =====
    episode_start = load_catch_model(model_path, hppo_agent, path_list['model_path_catch_PPO_h'])

    tai_episoid = load_tai_model(tai_agent, path_list['model_path_tai_PPO_h'], default_episode=tai_episoid)

    decision_episode = load_decision_model(decision_hppo_agent, path_list.get('model_path_decision_PPO_h'))

    # ===== 索引与计数（集中管理） =====
    episode_num = episode_start           # 抓取阶段起始轮次
    # 决策层起始轮次（从决策模型文件名恢复，若不存在则为0）
    decision_episode = decision_episode if 'decision_episode' in locals() else 0
    total_episode = decision_episode      # 总轮次计数
    success_catch = 0                     # 抓取成功次数
    catch_success = False                 # 跨episode标记：上一轮是否抓取成功

    # ===============================
    # 上层总训练循环（新增）
    # ===============================
    MAX_TOTAL_EPISODE = 3000

    env = Environment()  # 仍然只有一个 env

    while total_episode < MAX_TOTAL_EPISODE:

        print(f"\n==============================")
        print(f"🌍 Total Episode {total_episode}")
        print(f"==============================")

        # ---------- 上层决策 ----------
        # 修复：添加 imgs 参数
        d_steps = 0
        d_imgs = []  # 添加图像列表
        d_obs_img, d_obs_tensor = env.get_img(d_steps, d_imgs)  # 传入两个参数
        d_robot_state = env.get_robot_state()
        d_obs = (d_obs_tensor, d_robot_state)
        
        # 调试：打印输入形状和值范围
        print(f"📊 d_obs_tensor shape: {d_obs_tensor.shape}, range: [{d_obs_tensor.min():.3f}, {d_obs_tensor.max():.3f}]")
        print(f"📊 d_robot_state shape: {len(d_robot_state) if isinstance(d_robot_state, list) else d_robot_state.shape}")
        
        decision_dict = decision_hppo_agent.choose_action(
            obs=d_obs,
            x_graph=d_robot_state
        )

        # decision ∈ {0,1}
        decision = int(decision_dict['discrete_action'][0])
        print(f"上层决策 decision = {decision} (0=抓取, 1=爬梯)")


        if decision == 0:
            print("🟢 进入【抓取训练阶段】")
            # catch_success由上一轮保持，不重置（除非抬腿完成后）

            # 直接使用外层的episode_num，不用for循环
            log_writer_catch.add(episode_num=episode_num)
            print(f"<<<<<<<<<第{episode_num}周期")  # 打印当前周期
            env.reset()
            env.wait(500)  # 等待500ms
            imgs = []  # 初始化图像列表
            steps = 0  # 初始化步数
            return_all = 0  # 初始化总奖励
            obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
            # log_writer.add(obs_img=obs_img, steps=steps)
            robot_state = env.get_robot_state()
            # print(f'robot_state: {robot_state}')
            # print(f'robot_state_len: {len(robot_state)}')
            print("____________________")  # 打印初始状态
            # 记录上一次实际发送到环境的动作（用于离散=0时保持不变）
            prev_shoulder_action = 0.0
            prev_arm_action = 0.0
            prev_distance = None
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

                    dict = hppo_agent.choose_action(episode_num=episode_num,
                                                    obs=obs,
                                                    x_graph=robot_state)

                    d_action = dict['discrete_action']

                    action_shoulder = dict['continuous_action'][0]
                    action_arm = dict['continuous_action'][1]
                    log_prob_shoulder = dict['continuous_log_prob'][0]
                    log_prob_arm = dict['continuous_log_prob'][1]
                    value = dict['value']


                    d0 = float(d_action[0])
                    d1 = float(d_action[1])
                    cur_shoulder = float(action_shoulder.item())
                    cur_arm = float(action_arm.item())
                    # 若离散为0，则保持上一时刻指令
                    masked_shoulder = prev_shoulder_action if int(d0) == 0 else cur_shoulder
                    masked_arm = prev_arm_action if int(d1) == 0 else cur_arm
                    print(
                        f'第{episode_num}周期，第{steps}步, 离散数值：({int(d0)},{int(d1)}), 连续: ({action_shoulder.item():.4f},{action_arm.item():.4f}), 实际: ({masked_shoulder:.4f},{masked_arm:.4f})')


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
                    log_writer_catch.add_value_catch(value, value)
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

                    gps1, _, _, _, _ = env.print_gps()
                    # 安全检查：确保gps1有足够的元素
                    if len(gps1) < 3:
                        print(f"警告：gps1长度不足 ({len(gps1)} < 3)，使用默认值")
                        dx = 0.0
                        dy = 0.0
                    else:
                        dx = gps_goal[0] - gps1[1]
                        dy = gps_goal[1] - gps1[2]
                    current_distance = (dx ** 2 + dy ** 2) ** 0.5

                    # 距离变化奖励（鼓励靠近目标）
                    success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')
                    if prev_distance is not None:
                        distance_reward = (prev_distance - current_distance) * 15.0  # 增加距离奖励权重
                    else:
                        distance_reward = -current_distance  # 初始奖励

                        # 组合奖励
                        prev_distance = current_distance  # 更新

                    # 添加距离奖励：越接近目标奖励越高
                    proximity_reward = max(0, (0.5 - current_distance) * 5.0)  # 距离小于0.5时给额外奖励

                    # 动作程度奖惩
                    action_magnitude = (abs(cur_shoulder) + abs(cur_arm)) / 2.0
                    inactivity_penalty = -0.2 if action_magnitude < 0.05 else 0.0
                    if int(d0) == 0 and int(d1) == 0:
                        inactivity_penalty += -0.3
                    large_action_penalty = -0.05 * (abs(cur_shoulder) > 0.9 or abs(cur_arm) > 0.9)

                    # --- 合并奖励 ---
                    reward = distance_reward + proximity_reward + inactivity_penalty + large_action_penalty - 0.5 * steps

                    all_grasp_sensors = [
                        env.darwin.get_touch_sensor_value('grasp_L1'),
                        env.darwin.get_touch_sensor_value('grasp_L1_1'),
                        env.darwin.get_touch_sensor_value('grasp_L1_2'),
                        env.darwin.get_touch_sensor_value('grasp_R1'),
                        env.darwin.get_touch_sensor_value('grasp_R1_1'),
                        env.darwin.get_touch_sensor_value('grasp_R1_2')
                    ]
                    left_sensors = all_grasp_sensors[0:3]
                    right_sensors = all_grasp_sensors[3:6]
                    left_any = any(left_sensors)
                    right_any = any(right_sensors)
                    # 抓取成功：左右两侧都有传感器触发
                    success_flag1 = 1 if (left_any and right_any) else 0

                    if success_flag1 == 1:  # 抓到了
                        # 用你前面算好的 current_distance 即可
                        if current_distance <= 0.04:  # 4 cm 容忍
                            reward += 30
                            print("✅ 抓到目标梯级，发放大奖励！")
                            catch_success = True  # 记录抓取成功
                        else:
                            reward -= 15  # 抓错梯子，无大奖励
                            print("⚠️  抓到非目标梯级，无大奖励")
                    if done == 1 and steps < 6 and success_flag1 != 1:
                        print("错误抓取！给予较大惩罚！")
                        reward -= 10
                    if done == 1 and steps >= 6 and success_flag1 != 1:
                        print("错误抓取！给予较大惩罚！")
                        reward -= 10
                    if done == 1 and steps <= 2 and success_flag1 != 1:
                        print("因环境不稳定导致无效数据，跳过此步骤！！！")
                        break
                    reward -= steps * 0.5
                    return_all = return_all + reward

                    return_all = return_all + reward  # 总奖励为当前奖励加上之前的总奖励
                    steps += 1  # 步数加1
                    next_obs_img, next_obs_tensor = env.get_img(steps, imgs)  # 获取下一个图像和图像张量
                    next_obs = [next_obs_img, next_state]
                    # print('获取下一个状态更新完毕')
                    # 可以修改reward值让其训练速度加快
                    if good == 1:  # 如果good为1
                        hppo_agent.store_transition(
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
                    robot_state = env.get_robot_state()  # 获取机器人状态

                    obs_tensor = next_obs_tensor  # 更新图像张量
                    # if temp < 5000:  # 如果经验回放缓存小于3000
                    # episode_num = 0  # 计数器为0
                    if episode_num >= 0 and done == 1:  # 只有在buffer中存满了数据才会学习
                        if goal == 1:  # 如果达到目标
                            print("goal = 1")
                            save_path = os.path.join(catch_checkpoint_dir, f"catch_hppo_{episode_num}.ckpt")
                            checkpoint = {
                                'policy': hppo_agent.policy.state_dict(),
                                'optimizer_hppo': hppo_agent.optimizer.state_dict(),
                                'episode': episode_num
                            }
                            torch.save(checkpoint, save_path)
                        # print("11111111111111111111111111111111111111111-303")
                        # loss_shoulder = ppo_shoulder.learn(action_type='shoulder')
                        # print("22222222222222222222222222222222222222222-305")
                        # loss_arm = ppo_arm.learn(action_type='arm')
                        # # 学习离散HPPO
                        # loss_hppo = hppo_switch_catch.learn()
                        loss_d, loss_c = hppo_agent.learn()

                        loss1, loss2 = loss_d, loss_c

                        # print('loss_arm:', loss_arm)
                        # print('loss_shoulder:', loss_shoulder)
                        # print('loss_hppo:', loss_hppo)
                        print('loss_discrete:', loss1, 'loss_continuous:', loss2)
                        
                        # 分别记录三个智能体的loss值
                        log_writer_catch.add_loss_hppo_catch(loss1, loss2)
                        # 立即落盘，避免仅在回合结束保存导致当轮loss缺失
                        try:
                            log_writer_catch.save_catch(log_file_latest_catch)
                        except Exception as _e:
                            print(f"保存抓取loss到日志失败: {_e}")

                        if episode_num % 100 == 0 and episode_num != 0:  # 每100步保存一次模型
                            save_path = os.path.join(catch_checkpoint_dir, f"catch_hppo_{episode_num}.ckpt")
                            checkpoint = {
                                'policy': hppo_agent.policy.state_dict(),
                                'optimizer_hppo': hppo_agent.optimizer.state_dict(),
                                'episode': episode_num
                            }
                            torch.save(checkpoint, save_path)

                        log_writer_catch.add(return_all=return_all)
                        # 写入目标
                        log_writer_catch.add(goal=goal)

                    success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

                    if catch_flag == 1.0 or done == 1: # 如果抓取器状态为1.0或完成
                        env.wait(100)
                        imgs = []
                        steps = 0
                        episode_num = episode_num + 1
                        
                        # 【新增】如果这个episode中成功抓取了，则让decision重新判断
                        if success_flag1 == 1 and current_distance <= 0.04:
                            print("【抓取成功】保持机器人抓取状态，准备进行decision的下一步判断...")
                            env.wait(200)  # 等待机器人稳定
                            log_writer_catch.save_catch(log_file_latest_catch)
                            break  # 退出抓取循环
    
                        log_writer_catch.clear()
                        log_writer_catch.save_catch(log_file_latest_catch)
                        break
            log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
        else:
            print("🟢 进入【抬腿训练阶段】")
            # 只有抓取成功后才允许抬腿；未抓取成功则跳过本轮抬腿
            if not catch_success:
                print("⚠️ 未检测到抓取成功，本轮跳过抬腿训练。")
                continue
            # if success_flag1 == 1:
            #     success_catch += 1
            #     log_writer_catch.add(success_catch=success_catch)
            #     print("success_catch:", success_catch)
            #     print("抓取成功，开始抬腿训练...")
            #     total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(existing_env=env, total_episode=total_episode, episode=tai_episoid,
                            log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai,
                            catch_success=catch_success, tai_agent=tai_agent)
            tai_episoid += 1
            
            # 抬腿执行完毕，重置环境和抓取标记，准备下一个完整循环
            catch_success = False
            env.reset()
            env.wait(500)

        # ===== 决策奖励计算（基于状态判断是否正确） =====
        decision_reward = 0.0
        if decision == 0:  # 决策选择抓取
            if catch_success:
                # 已经抓取成功还选择抓取，决策错误！
                decision_reward = -15.0
                print("❌ 决策错误：已抓取成功还选择抓取，惩罚-15.0")
            else:
                # 未抓取时选择抓取，决策正确
                decision_reward = 5.0
                print("✅ 决策正确：未抓取状态选择抓取，奖励+5.0")
        else:  # decision == 1，决策选择抬腿
            if catch_success:
                # 已抓取成功且选择抬腿，决策正确
                decision_reward = 10.0
                print("✅ 决策正确：已抓取状态选择抬腿，奖励+10.0")
            else:
                # 未抓取却选择抬腿，决策错误
                decision_reward = -10.0
                print("❌ 决策错误：未抓取状态选择抬腿，惩罚-10.0")
        
        # 记录决策日志
        log_writer_decision.add(episode_num=total_episode)
        log_writer_decision.add(decision=decision)
        log_writer_decision.add(decision_reward=decision_reward)
        log_writer_decision.add(catch_success=int(catch_success))
        
        # 决策智能体需要 state 包含 (x, state, x_graph)，将机器人状态复用为图输入以满足长度要求
        decision_state = (d_obs_tensor, d_robot_state, d_robot_state)
        decision_hppo_agent.store_transition(
            state=decision_state,
            action=decision,
            reward=decision_reward,  # 使用计算得到的奖励
            next_state=None,
            done=True,
            value=decision_dict['value'],
            log_prob=decision_dict['discrete_log_prob']
        )

        # 记录决策的value值
        log_writer_decision.add(decision_value=decision_dict['value'])
        
        # 训练决策智能体并获取loss（hppo只返回一个total_loss）
        decision_loss = decision_hppo_agent.learn()
        
        # 记录决策的loss值
        log_writer_decision.add(decision_loss=decision_loss)
        
        # 保存决策日志
        log_writer_decision.save_catch(log_file_latest_decision)

        # 定期保存决策智能体
        if total_episode % 50 == 0:
            dec_path = os.path.join(decision_checkpoint_dir, f"decision_hppo_{total_episode}.ckpt")
            dec_ckpt = {
                'policy': decision_hppo_agent.policy.state_dict(),
                'optimizer': decision_hppo_agent.optimizer.state_dict(),
                'episode': total_episode
            }
            torch.save(dec_ckpt, dec_path)

        total_episode += 1

    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env