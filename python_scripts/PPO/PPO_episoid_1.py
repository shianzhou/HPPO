# 测试
import torch
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
from python_scripts.PPO.hppo_01 import HPPO as hppo
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal
from python_scripts.PPO_Log_write import Log_write


# ===== 路径与文件工具函数（统一管理） =====
import os
import glob
import re


# ===== 方案A: 训练管理器（同步三个模型的更新频率） =====
class TrainingManager:
    """
    单智能体学习节奏管理。
    核心思想：抓取和抬腿都先积累经验，再按统一间隔学习。
    """
    def __init__(self):
        self.shared_episodes = 0
        
        # 统一学习间隔，避免同一智能体在不同阶段过于频繁更新
        self.shared_learn_interval = 3
        
        print("【训练管理器初始化】")
        print(f"  单智能体学习间隔: {self.shared_learn_interval}个episodes")
    
    def should_learn_shared(self) -> bool:
        """决定是否执行单智能体学习"""
        result = (self.shared_episodes % self.shared_learn_interval == 0) and (self.shared_episodes > 0)
        return result
    
    def increment_shared(self):
        """单智能体episode计数加1"""
        self.shared_episodes += 1
    
    def get_status(self) -> str:
        """获取当前训练状态"""
        return f"[TrainingManager] Shared:{self.shared_episodes}"


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

def _latest_single_ckpt(dir_path: str):
    """加载目录中最新的 single_hppo_*.ckpt。"""
    files = glob.glob(os.path.join(dir_path, "single_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(f: str) -> int:
        b = os.path.basename(f)
        m_new = re.search(r"single_hppo_(\d+)\.ckpt$", b)
        return int(m_new.group(1)) if m_new else -1

    selected = max(files, key=_num)
    return selected, _num(selected)

# ===== 模型加载工具函数（提炼提高可读性） =====
def load_single_model(model_path: str, hppo_agent, ckpt_dir: str) -> int:
    """加载单智能体模型，优先指定路径；否则自动加载目录最新。"""
    episode_start = 0
    if model_path:
        try:
            ckpt = torch.load(model_path)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print(f"从指定模型加载: {model_path}，单智能体模型加载成功！")
                try:
                    episode_start = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
                    print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
                except Exception:
                    pass
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print(f"从指定模型加载: {model_path}，单智能体模型加载成功！")
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
        return episode_start

    # 未指定路径，查找目录最新
    selected_model, episode_start = _latest_single_ckpt(ckpt_dir)
    if selected_model:
        try:
            ckpt = torch.load(selected_model)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print("单智能体模型加载成功！")
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print("单智能体模型加载成功！")
        except Exception as e:
            print(f"单智能体模型加载失败: {e}")
            episode_start = 0
    else:
        print("未找到已保存的单智能体模型，从头开始训练")
        episode_start = 0
    return episode_start


def PPO_episoid_1(model_path=None, max_steps_per_episode=5):
    # ===== 训练管理器初始化（方案A核心） =====
    training_manager = TrainingManager()
    
    # ===== 智能体实例化（单智能体：6离散 + 5连续） =====
    # 离散映射: 0=决策, 1-2=抓取, 3-5=抬腿
    # 连续映射: 0-1=抓取, 2-4=踩踏(抬腿)
    hppo_agent = hppo(
        num_servos=6,
        node_num=19,
        env_information=None
    )

    # ===== 日志写入器（单文件） =====
    log_writer = Log_write()

    # ===== 基础计数 =====
    tai_episoid = 1

    # ===== 模型保存目录（统一，使用配置的新路径） =====
    catch_checkpoint_dir = path_list['model_path_catch_PPO_h']
    _ensure_dir(catch_checkpoint_dir)

    # ===== 日志文件（自动递增编号，单文件） =====
    _ensure_dir(path_list['single_log_path_PPO'])
    log_file_latest_single = _next_log_file(path_list['single_log_path_PPO'], 'single_log')
    print(f"将使用单智能体统一日志: {log_file_latest_single}")

    # ===== 模型加载（函数化） =====
    episode_start = load_single_model(model_path, hppo_agent, path_list['model_path_catch_PPO_h'])

    # ===== 索引与计数（集中管理） =====
    episode_num = episode_start           # 抓取阶段起始轮次
    total_episode = 0                     # 总轮次计数
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
        env.reset()
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
        
        decision_dict = hppo_agent.choose_action(
            episode_num=total_episode,
            obs=d_obs,
            x_graph=d_robot_state
        )

        # decision 来自第1个离散输出
        decision = int(decision_dict['discrete_action'][0])
        print(f"上层决策 decision = {decision} (0=抓取, 1=爬梯)")
        decision_catch_success = catch_success
        decision_reward = (5.0 if (decision == 0 and not decision_catch_success) else
                   -15.0 if (decision == 0 and decision_catch_success) else
                   10.0 if (decision == 1 and decision_catch_success) else
                   -10.0)


        if decision == 0:
            print("🟢 进入【抓取训练阶段】")
            # catch_success由上一轮保持，不重置（除非抬腿完成后）

            # 直接使用外层的episode_num，不用for循环
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
            catch_loss_discrete = 0
            catch_loss_continuous = 0
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


                    d1 = float(d_action[1])
                    d2 = float(d_action[2])
                    cur_shoulder = float(action_shoulder.item())
                    cur_arm = float(action_arm.item())
                    # 若离散为0，则保持上一时刻指令
                    masked_shoulder = prev_shoulder_action if int(d1) == 0 else cur_shoulder
                    masked_arm = prev_arm_action if int(d2) == 0 else cur_arm
                    print(
                        f'第{episode_num}周期，第{steps}步, 离散数值：(决策{int(d_action[0])},抓取{int(d1)}/{int(d2)}), 连续: ({action_shoulder.item():.4f},{action_arm.item():.4f}), 实际: ({masked_shoulder:.4f},{masked_arm:.4f})')


                    gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
                    if steps >= 19:  # 如果步数大于等于19
                        catch_flag = 1.0  # 抓取器状态为1.0
                    else:
                        catch_flag = 0.0  # 抓取器状态为0.0
                    img_name = "img" + str(steps) + ".png"  # 图像名称
                    # print("action:", a)
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
                    if int(d1) == 0 and int(d2) == 0:
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
                            save_path = os.path.join(catch_checkpoint_dir, f"single_hppo_{episode_num}.ckpt")
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
                        
                        # 单智能体统一学习时机
                        training_manager.increment_shared()
                        if training_manager.should_learn_shared():
                            loss_d, loss_c = hppo_agent.learn()
                            catch_loss_discrete, catch_loss_continuous = loss_d, loss_c
                            loss1, loss2 = loss_d, loss_c
                            print(f'【单智能体学习-抓取阶段】{training_manager.get_status()} | loss_discrete: {loss1:.6f}, loss_continuous: {loss2:.6f}')
                        else:
                            # 累积经验但不学习
                            print(f'【单智能体累积经验-抓取阶段】{training_manager.get_status()}')
                            loss1, loss2 = 0, 0

                        if episode_num % 100 == 0 and episode_num != 0:  # 每100步保存一次模型
                            save_path = os.path.join(catch_checkpoint_dir, f"single_hppo_{episode_num}.ckpt")
                            checkpoint = {
                                'policy': hppo_agent.policy.state_dict(),
                                'optimizer_hppo': hppo_agent.optimizer.state_dict(),
                                'episode': episode_num
                            }
                            torch.save(checkpoint, save_path)

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
                            break  # 退出抓取循环
    
                        break
            log_writer.log_cycle(
                log_file_latest_single,
                episode_num=total_episode,
                action_type='抓取',
                decision_reward=decision_reward,
                catch_reward=return_all,
                total_reward=return_all + decision_reward,
                loss_discrete=catch_loss_discrete,
                loss_continuous=catch_loss_continuous,
                total_episode_num=total_episode,
                phase_episode_num=episode_num,
            )
        else:
            print("🟢 进入【抬腿训练阶段】")
            # 只有抓取成功后才允许抬腿；未抓取成功则跳过本轮抬腿
            if not catch_success:
                print("⚠️ 未检测到抓取成功，本轮跳过抬腿训练。")
                log_writer.log_cycle(
                    log_file_latest_single,
                    episode_num=total_episode,
                    action_type='抬腿-跳过',
                    decision_reward=decision_reward,
                    total_reward=decision_reward,
                    loss_discrete=0,
                    loss_continuous=0,
                    total_episode_num=total_episode,
                    phase_episode_num=tai_episoid,
                )
                total_episode += 1
                continue
            # if success_flag1 == 1:
            #     success_catch += 1
            #     log_writer_catch.add(success_catch=success_catch)
            #     print("success_catch:", success_catch)
            #     print("抓取成功，开始抬腿训练...")
            #     total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(existing_env=env, total_episode=total_episode, episode=tai_episoid,
                            log_writer_tai=log_writer, log_file_latest_tai=log_file_latest_single,
                            catch_success=catch_success, hppo_agent=hppo_agent, training_manager=training_manager,
                            decision_reward=decision_reward,
                            discrete_indices=(3, 4, 5), continuous_indices=(2, 3, 4))
            tai_episoid += 1
            
            # 抬腿执行完毕，重置环境和抓取标记，准备下一个完整循环
            catch_success = False
            env.reset()
            env.wait(500)

        total_episode += 1

    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env