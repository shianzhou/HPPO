import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal
from torch.distributions import Categorical
from python_scripts.Project_config import device

class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, num_servos, node_num):
        super().__init__()
        self.num_servos = num_servos
        self.node_num = node_num
        # 动作映射固定为：离散6维(1决策+2抓取+3踩踏)，连续5维(2抓取+3踩踏)
        self.num_discrete_actions = 6
        self.num_continuous_actions = 5
        # 图像特征提取
        self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2), padding=1)
        self.fc0 = nn.Linear(6272, 6000)
        self.fc1 = nn.Linear(6000, 100)
        self.fc2 = nn.Linear(20, 100)  # 修改输入维度从4改为20
        self.fc3 = nn.Linear(100, 100)
        # 图神经网络部分（可选，简化版）
        self.fc_graph = nn.Linear(20, 100)  # 修改输入维度从node_num改为20
        # 共享特征层
        self.fc4 = nn.Linear(300, 200)
        # 3个离散策略头：决策(1维)、抓取门控(2维)、踩踏门控(3维)
        self.discrete_decision_head = nn.Linear(200, 2)
        self.discrete_catch_head = nn.Linear(200, 2 * 2)
        self.discrete_tai_head = nn.Linear(200, 3 * 2)

        # 2个连续策略头：抓取参数(2维)、踩踏参数(3维)
        self.continuous_catch = nn.Sequential(
            nn.Linear(200, 2),
            nn.Tanh()
        )
        self.continuous_tai = nn.Sequential(
            nn.Linear(200, 3),
            nn.Tanh()
        )
        self.actor_log_sigma_catch = nn.Parameter(torch.zeros(2) * 0.1)
        self.actor_log_sigma_tai = nn.Parameter(torch.zeros(3) * 0.1)

        # Critic头
        self.critic = nn.Linear(200, 1)

    def forward(self, x, state, x_graph):
        # 图像特征（兼容单样本与批量输入）
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.dim() == 2:
            # [H,W] -> [1,1,H,W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[0] == 1:
                # [C,H,W] -> [1,C,H,W]
                x = x.unsqueeze(0)
            else:
                # [B,H,W] -> [B,1,H,W]
                x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Unsupported image tensor shape: {tuple(x.shape)}")
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc0(x)
        x = self.fc1(x)

        # 状态特征
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = self.fc2(state)
        state = self.fc3(state)

        # 图特征（简化为全连接）
        x_graph = torch.as_tensor(x_graph, dtype=torch.float32, device=device)
        if x_graph.dim() == 1:
            x_graph = x_graph.unsqueeze(0)
        x_graph = self.fc_graph(x_graph)

        # 融合
        state_x = torch.cat((x, state, x_graph), dim=-1)


        features = self.fc4(state_x)
        decision_logits = self.discrete_decision_head(features).view(-1, 2)
        grab_logits = self.discrete_catch_head(features).view(-1, 2, 2)
        step_logits = self.discrete_tai_head(features).view(-1, 3, 2)

        grab_mu = self.continuous_catch(features).view(-1, 2)
        step_mu = self.continuous_tai(features).view(-1, 3)
        value = self.critic(features).view(-1)

        return decision_logits, grab_logits, step_logits, grab_mu, step_mu, value

class HPPO:
    def __init__(self, num_servos, node_num, env_information=None ):
        self.num_servos = num_servos
        self.node_num = node_num
        self.env_information = env_information
        self.device = device
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_update_epochs = 3  # 【修改】从10轮减少到3轮，防止梯度聚集
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # 学习率设置 - 与其他PPO保持一致
        self.lr = 5e-5  # 【修改】从2e-4降低到5e-5，进一步降低学习率避免参数剧烈波动
        self.lr_decay = 0.995  # 学习率衰减
        
        # 网络
        self.policy = MultiDiscreteActorCritic(num_servos, node_num).to(device)
        
        # 单优化器：统一更新共享层 + 所有分支头 + value 头
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        # 学习率调度器 - 添加学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        # Advantage 调试输出开关；打开后每次 learn 都会打印分布统计
        self.print_advantage_stats = False
        # 轨迹缓存
        self.states = []

        self.decisions = []
        self.decision_log_probs = []

        self.grab_discrete_actions = []
        self.grab_discrete_log_probs = []
        self.step_discrete_actions = []
        self.step_discrete_log_probs = []

        self.grab_continuous_actions = []
        self.grab_continuous_log_probs = []
        self.step_continuous_actions = []
        self.step_continuous_log_probs = []

        self.grab_success_flags = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.dones = []
        self.last_learn_stats = {
            'loss_decision': None,
            'loss_grab_discrete': None,
            'loss_step_discrete': None,
            'loss_grab_continuous': None,
            'loss_step_continuous': None,
            'loss_value': None,
            'loss_total': None,
            'grab_mask_mean': None,
            'step_mask_mean': None,
            'adv_raw_mean': None,
            'adv_raw_std': None,
            'adv_raw_max': None,
            'adv_raw_min': None,
            'adv_norm_mean': None,
            'adv_norm_std': None,
            'adv_norm_max': None,
            'adv_norm_min': None,
            'ratio_decision_mean': None,
            'ratio_decision_std': None,
            'ratio_decision_p05': None,
            'ratio_decision_p95': None,
            'clipfrac_decision': None,
            'approx_kl_decision': None,
            'ratio_grab_mean': None,
            'clipfrac_grab': None,
            'approx_kl_grab': None,
            'ratio_step_mean': None,
            'clipfrac_step': None,
            'approx_kl_step': None,
            'returns_mean': None,
            'returns_std': None,
            'values_mean': None,
            'values_std': None,
            'explained_variance': None,
        }

    def _build_distributions(self, decision_logits, grab_logits, step_logits, grab_mu, step_mu):
        decision_dist = Categorical(logits=decision_logits)
        grab_dist = Categorical(logits=grab_logits)
        step_dist = Categorical(logits=step_logits)

        grab_sigma = torch.exp(self.policy.actor_log_sigma_catch).expand_as(grab_mu)
        step_sigma = torch.exp(self.policy.actor_log_sigma_tai).expand_as(step_mu)
        grab_cont_dist = Normal(grab_mu, grab_sigma)
        step_cont_dist = Normal(step_mu, step_sigma)
        return decision_dist, grab_dist, step_dist, grab_cont_dist, step_cont_dist


    def choose_action(self, episode_num, obs, x_graph):
        with torch.no_grad():
            decision_logits, grab_logits, step_logits, grab_mu, step_mu, value = self.policy(
                x=obs[0],
                state=obs[1],
                x_graph=x_graph
            )

            decision_dist, grab_dist, step_dist, grab_cont_dist, step_cont_dist = self._build_distributions(
                decision_logits, grab_logits, step_logits, grab_mu, step_mu
            )

            decision = decision_dist.sample().squeeze(0)
            grab_discrete = grab_dist.sample().squeeze(0)
            step_discrete = step_dist.sample().squeeze(0)

            grab_continuous = torch.clamp(grab_cont_dist.sample().squeeze(0), min=-1.0, max=1.0)
            step_continuous = torch.clamp(step_cont_dist.sample().squeeze(0), min=-1.0, max=1.0)

            decision_log_prob = decision_dist.log_prob(decision)
            grab_discrete_log_prob = grab_dist.log_prob(grab_discrete)
            step_discrete_log_prob = step_dist.log_prob(step_discrete)
            grab_continuous_log_prob = grab_cont_dist.log_prob(grab_continuous).sum(-1)
            step_continuous_log_prob = step_cont_dist.log_prob(step_continuous).sum(-1)

            # 分布保留了 batch 维，这里统一去掉，确保后续拼接维度一致
            if decision_log_prob.dim() > 0:
                decision_log_prob = decision_log_prob.squeeze(0)
            if grab_discrete_log_prob.dim() > 1:
                grab_discrete_log_prob = grab_discrete_log_prob.squeeze(0)
            if step_discrete_log_prob.dim() > 1:
                step_discrete_log_prob = step_discrete_log_prob.squeeze(0)

            grab_continuous_log_prob_vec = grab_cont_dist.log_prob(grab_continuous)
            step_continuous_log_prob_vec = step_cont_dist.log_prob(step_continuous)
            if grab_continuous_log_prob_vec.dim() > 1:
                grab_continuous_log_prob_vec = grab_continuous_log_prob_vec.squeeze(0)
            if step_continuous_log_prob_vec.dim() > 1:
                step_continuous_log_prob_vec = step_continuous_log_prob_vec.squeeze(0)

            discrete_action = torch.cat((decision.view(1), grab_discrete, step_discrete), dim=0)
            continuous_action = torch.cat((grab_continuous, step_continuous), dim=0)

            discrete_log_prob = torch.cat(
                (decision_log_prob.view(1), grab_discrete_log_prob, step_discrete_log_prob),
                dim=0
            )
            continuous_log_prob = torch.cat(
                (
                    grab_continuous_log_prob_vec,
                    step_continuous_log_prob_vec,
                ),
                dim=0
            )

            if isinstance(value, torch.Tensor):
                value_scalar = value.item()  # 提取浮点数
            else:
                value_scalar = value  # 已经是浮点数，直接使用


            action_dict = {
                'decision': int(decision.item()),
                'decision_log_prob': float(decision_log_prob.item()),
                'grab_discrete': grab_discrete.cpu().numpy(),
                'grab_discrete_log_prob': grab_discrete_log_prob.cpu().numpy(),
                'step_discrete': step_discrete.cpu().numpy(),
                'step_discrete_log_prob': step_discrete_log_prob.cpu().numpy(),
                'grab_continuous': grab_continuous.cpu().numpy(),
                'grab_continuous_log_prob': float(grab_continuous_log_prob.item()),
                'step_continuous': step_continuous.cpu().numpy(),
                'step_continuous_log_prob': float(step_continuous_log_prob.item()),

                # 兼容旧调用方
                'discrete_action': discrete_action.cpu().numpy(),
                'continuous_action': continuous_action.cpu().numpy(),
                'discrete_log_prob': discrete_log_prob.cpu().numpy(),
                'continuous_log_prob': continuous_log_prob.cpu().numpy(),
                'value':  value_scalar  # 状态价值是标量
            }
            return action_dict

    # def store_transition(self, state, discrete_action,continuous_action, reward, next_state, done, value, log_prob):
    #     self.states.append(state)
    #     self.discrete_actions.append(discrete_action)
    #     self.continuous_actions.append(continuous_action)
    #     self.rewards.append(reward)
    #     self.next_states.append(next_state)
    #     self.values.append(value)
    #     self.log_probs.append(log_prob)
    #     self.dones.append(done)
    def store_transition(self, state, discrete_action, continuous_action, reward, next_state, done, value,
                         discrete_log_prob, continuous_log_prob, decision=None, decision_log_prob=None,
                         grab_discrete=None, grab_discrete_log_prob=None, step_discrete=None,
                         step_discrete_log_prob=None, grab_continuous=None,
                         grab_continuous_log_prob=None, step_continuous=None,
                         step_continuous_log_prob=None, grab_success=True):
        discrete_action = np.asarray(discrete_action)
        continuous_action = np.asarray(continuous_action)
        discrete_log_prob = np.asarray(discrete_log_prob)

        if decision is None:
            decision = int(discrete_action[0])
        if decision_log_prob is None:
            decision_log_prob = float(discrete_log_prob[0])

        if grab_discrete is None:
            grab_discrete = discrete_action[1:3]
        if grab_discrete_log_prob is None:
            grab_discrete_log_prob = discrete_log_prob[1:3]

        if step_discrete is None:
            step_discrete = discrete_action[3:6]
        if step_discrete_log_prob is None:
            step_discrete_log_prob = discrete_log_prob[3:6]

        if grab_continuous is None:
            grab_continuous = continuous_action[0:2]
        if step_continuous is None:
            step_continuous = continuous_action[2:5]

        continuous_log_prob = np.asarray(continuous_log_prob)
        if grab_continuous_log_prob is None:
            grab_continuous_log_prob = float(np.sum(continuous_log_prob[0:2]))
        if step_continuous_log_prob is None:
            step_continuous_log_prob = float(np.sum(continuous_log_prob[2:5]))

        self.states.append(state)
        self.decisions.append(int(decision))
        self.decision_log_probs.append(float(decision_log_prob))

        self.grab_discrete_actions.append(np.asarray(grab_discrete, dtype=np.int64))
        self.grab_discrete_log_probs.append(np.asarray(grab_discrete_log_prob, dtype=np.float32))
        self.step_discrete_actions.append(np.asarray(step_discrete, dtype=np.int64))
        self.step_discrete_log_probs.append(np.asarray(step_discrete_log_prob, dtype=np.float32))

        self.grab_continuous_actions.append(np.asarray(grab_continuous, dtype=np.float32))
        self.grab_continuous_log_probs.append(float(grab_continuous_log_prob))
        self.step_continuous_actions.append(np.asarray(step_continuous, dtype=np.float32))
        self.step_continuous_log_probs.append(float(step_continuous_log_prob))

        self.grab_success_flags.append(float(grab_success))
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.dones.append(done)

    def calculate_advantages(self):
        advantages = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(device)

    # def get_value(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     with torch.no_grad():
    #         value = self.critic(state)
    #     return value.cpu().data.numpy().squeeze(0)

    def _clear_buffer(self):
        self.states = []

        self.decisions = []
        self.decision_log_probs = []

        self.grab_discrete_actions = []
        self.grab_discrete_log_probs = []
        self.step_discrete_actions = []
        self.step_discrete_log_probs = []

        self.grab_continuous_actions = []
        self.grab_continuous_log_probs = []
        self.step_continuous_actions = []
        self.step_continuous_log_probs = []

        self.grab_success_flags = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.dones = []


    def learn(self):
        if len(self.states) < 32:  # 使用定义的batch_size
            self.last_learn_stats = {
                'loss_decision': None,
                'loss_grab_discrete': None,
                'loss_step_discrete': None,
                'loss_grab_continuous': None,
                'loss_step_continuous': None,
                'loss_value': None,
                'loss_total': None,
                'grab_mask_mean': None,
                'step_mask_mean': None,
                'adv_raw_mean': None,
                'adv_raw_std': None,
                'adv_raw_max': None,
                'adv_raw_min': None,
                'adv_norm_mean': None,
                'adv_norm_std': None,
                'adv_norm_max': None,
                'adv_norm_min': None,
                'ratio_decision_mean': None,
                'ratio_decision_std': None,
                'ratio_decision_p05': None,
                'ratio_decision_p95': None,
                'clipfrac_decision': None,
                'approx_kl_decision': None,
                'ratio_grab_mean': None,
                'clipfrac_grab': None,
                'approx_kl_grab': None,
                'ratio_step_mean': None,
                'clipfrac_step': None,
                'approx_kl_step': None,
                'returns_mean': None,
                'returns_std': None,
                'values_mean': None,
                'values_std': None,
                'explained_variance': None,
            }
            return 0,0

        # 计算优势函数和回报（当前实现正确，无需修改）
        raw_advantages = self.calculate_advantages()
        returns = raw_advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

        adv_raw_mean = raw_advantages.mean().item()
        adv_raw_std = raw_advantages.std().item()
        adv_raw_max = raw_advantages.max().item()
        adv_raw_min = raw_advantages.min().item()
        returns_mean = returns.mean().item()
        returns_std = returns.std().item()

        # 转换为张量并移动到设备
        batch_states = self.states  # 列表，每个元素是状态元组
        batch_x = torch.as_tensor(np.asarray([s[0] for s in batch_states]), dtype=torch.float32, device=self.device)
        batch_state_vec = torch.as_tensor(np.asarray([s[1] for s in batch_states]), dtype=torch.float32, device=self.device)
        batch_graph = torch.as_tensor(np.asarray([s[2] for s in batch_states]), dtype=torch.float32, device=self.device)
        # 先汇总为 numpy 数组，再转 tensor，避免 list[np.ndarray] 的慢路径告警
        batch_decisions = torch.as_tensor(np.asarray(self.decisions), dtype=torch.long, device=self.device)
        batch_decision_log_probs = torch.as_tensor(np.asarray(self.decision_log_probs), dtype=torch.float32, device=self.device)

        batch_grab_discrete_actions = torch.as_tensor(np.asarray(self.grab_discrete_actions), dtype=torch.long, device=self.device)
        batch_grab_discrete_log_probs = torch.as_tensor(np.asarray(self.grab_discrete_log_probs), dtype=torch.float32, device=self.device)
        batch_step_discrete_actions = torch.as_tensor(np.asarray(self.step_discrete_actions), dtype=torch.long, device=self.device)
        batch_step_discrete_log_probs = torch.as_tensor(np.asarray(self.step_discrete_log_probs), dtype=torch.float32, device=self.device)

        batch_grab_continuous_actions = torch.as_tensor(np.asarray(self.grab_continuous_actions), dtype=torch.float32, device=self.device)
        batch_grab_continuous_log_probs = torch.as_tensor(np.asarray(self.grab_continuous_log_probs), dtype=torch.float32, device=self.device)
        batch_step_continuous_actions = torch.as_tensor(np.asarray(self.step_continuous_actions), dtype=torch.float32, device=self.device)
        batch_step_continuous_log_probs = torch.as_tensor(np.asarray(self.step_continuous_log_probs), dtype=torch.float32, device=self.device)

        batch_grab_success = torch.as_tensor(np.asarray(self.grab_success_flags), dtype=torch.float32, device=self.device)

        # 标准化优势
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)

        adv_norm_mean = advantages.mean().item()
        adv_norm_std = advantages.std().item()
        adv_norm_max = advantages.max().item()
        adv_norm_min = advantages.min().item()

        if self.print_advantage_stats:
            print(
                f"[ADV raw] mean={adv_raw_mean:.6f}, std={adv_raw_std:.6f}, max={adv_raw_max:.6f}, min={adv_raw_min:.6f}"
            )
            print(
                f"[ADV norm] mean={adv_norm_mean:.6f}, std={adv_norm_std:.6f}, max={adv_norm_max:.6f}, min={adv_norm_min:.6f}"
            )

        loss_continuous = 0
        loss_discrete = 0
        loss_decision_sum = 0
        loss_grab_discrete_sum = 0
        loss_step_discrete_sum = 0
        loss_grab_continuous_sum = 0
        loss_step_continuous_sum = 0
        loss_value_sum = 0
        loss_total_sum = 0
        grab_mask_mean_sum = 0
        step_mask_mean_sum = 0
        ratio_decision_mean_sum = 0
        ratio_decision_std_sum = 0
        ratio_decision_p05_sum = 0
        ratio_decision_p95_sum = 0
        clipfrac_decision_sum = 0
        approx_kl_decision_sum = 0
        ratio_grab_mean_sum = 0
        clipfrac_grab_sum = 0
        approx_kl_grab_sum = 0
        ratio_step_mean_sum = 0
        clipfrac_step_sum = 0
        approx_kl_step_sum = 0
        values_mean_sum = 0
        values_std_sum = 0
        explained_variance_sum = 0

        def masked_mean(x, mask):
            denom = torch.clamp(mask.sum(), min=1.0)
            return ((x * mask).sum() / denom).item()

        for _ in range(self.policy_update_epochs):
            decision_logits, grab_logits, step_logits, grab_mu, step_mu, all_values = self.policy(
                x=batch_x,
                state=batch_state_vec,
                x_graph=batch_graph
            )
            decision_dist, grab_dist, step_dist, grab_cont_dist, step_cont_dist = self._build_distributions(
                decision_logits, grab_logits, step_logits, grab_mu, step_mu
            )

            new_decision_log_probs = decision_dist.log_prob(batch_decisions)
            new_grab_discrete_log_probs = grab_dist.log_prob(batch_grab_discrete_actions)
            new_step_discrete_log_probs = step_dist.log_prob(batch_step_discrete_actions)
            new_grab_cont_log_probs = grab_cont_dist.log_prob(batch_grab_continuous_actions).sum(-1)
            new_step_cont_log_probs = step_cont_dist.log_prob(batch_step_continuous_actions).sum(-1)

            decision_mask = torch.ones_like(advantages)
            grab_mask = (batch_decisions == 0).float()
            step_mask = ((batch_decisions == 1).float() * batch_grab_success)

            old_grab_discrete_log_probs = batch_grab_discrete_log_probs.sum(-1)
            old_step_discrete_log_probs = batch_step_discrete_log_probs.sum(-1)

            ratio_decision = torch.exp(new_decision_log_probs - batch_decision_log_probs)
            ratio_grab = torch.exp(new_grab_discrete_log_probs.sum(-1) - old_grab_discrete_log_probs)
            ratio_step = torch.exp(new_step_discrete_log_probs.sum(-1) - old_step_discrete_log_probs)

            ratio_decision_mean_sum += ratio_decision.mean().item()
            ratio_decision_std_sum += ratio_decision.std().item()
            ratio_decision_p05_sum += torch.quantile(ratio_decision, 0.05).item()
            ratio_decision_p95_sum += torch.quantile(ratio_decision, 0.95).item()
            clipfrac_decision_sum += (torch.abs(ratio_decision - 1.0) > self.clip_ratio).float().mean().item()
            approx_kl_decision_sum += (batch_decision_log_probs - new_decision_log_probs).mean().item()

            ratio_grab_mean_sum += masked_mean(ratio_grab, grab_mask)
            clipfrac_grab_sum += masked_mean((torch.abs(ratio_grab - 1.0) > self.clip_ratio).float(), grab_mask)
            approx_kl_grab_sum += masked_mean(old_grab_discrete_log_probs - new_grab_discrete_log_probs.sum(-1), grab_mask)

            ratio_step_mean_sum += masked_mean(ratio_step, step_mask)
            clipfrac_step_sum += masked_mean((torch.abs(ratio_step - 1.0) > self.clip_ratio).float(), step_mask)
            approx_kl_step_sum += masked_mean(old_step_discrete_log_probs - new_step_discrete_log_probs.sum(-1), step_mask)

            def masked_ppo_loss(new_lp, old_lp, adv, mask):
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                loss_each = -torch.min(surr1, surr2)
                denom = torch.clamp(mask.sum(), min=1.0)
                return (loss_each * mask).sum() / denom

            decision_loss = masked_ppo_loss(new_decision_log_probs, batch_decision_log_probs, advantages, decision_mask)
            grab_discrete_loss = masked_ppo_loss(new_grab_discrete_log_probs.sum(-1), old_grab_discrete_log_probs, advantages, grab_mask)
            step_discrete_loss = masked_ppo_loss(new_step_discrete_log_probs.sum(-1), old_step_discrete_log_probs, advantages, step_mask)
            grab_continuous_loss = masked_ppo_loss(new_grab_cont_log_probs, batch_grab_continuous_log_probs, advantages, grab_mask)
            step_continuous_loss = masked_ppo_loss(new_step_cont_log_probs, batch_step_continuous_log_probs, advantages, step_mask)

            value_loss = ((all_values - returns) ** 2).mean()

            values_mean_sum += all_values.mean().item()
            values_std_sum += all_values.std().item()
            var_returns = torch.var(returns)
            explained_variance = 1.0 - torch.var(returns - all_values) / (var_returns + 1e-8)
            explained_variance_sum += explained_variance.item()

            total_loss = (
                decision_loss +
                grab_discrete_loss +
                grab_continuous_loss +
                step_discrete_loss +
                step_continuous_loss +
                self.value_coef * value_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            loss_discrete += (decision_loss + grab_discrete_loss + step_discrete_loss).item()
            loss_continuous += (grab_continuous_loss + step_continuous_loss + self.value_coef * value_loss).item()
            loss_decision_sum += decision_loss.item()
            loss_grab_discrete_sum += grab_discrete_loss.item()
            loss_step_discrete_sum += step_discrete_loss.item()
            loss_grab_continuous_sum += grab_continuous_loss.item()
            loss_step_continuous_sum += step_continuous_loss.item()
            loss_value_sum += value_loss.item()
            loss_total_sum += total_loss.item()
            grab_mask_mean_sum += grab_mask.mean().item()
            step_mask_mean_sum += step_mask.mean().item()

        self.last_learn_stats = {
            'loss_decision': loss_decision_sum / self.policy_update_epochs,
            'loss_grab_discrete': loss_grab_discrete_sum / self.policy_update_epochs,
            'loss_step_discrete': loss_step_discrete_sum / self.policy_update_epochs,
            'loss_grab_continuous': loss_grab_continuous_sum / self.policy_update_epochs,
            'loss_step_continuous': loss_step_continuous_sum / self.policy_update_epochs,
            'loss_value': loss_value_sum / self.policy_update_epochs,
            'loss_total': loss_total_sum / self.policy_update_epochs,
            'grab_mask_mean': grab_mask_mean_sum / self.policy_update_epochs,
            'step_mask_mean': step_mask_mean_sum / self.policy_update_epochs,
            'adv_raw_mean': adv_raw_mean,
            'adv_raw_std': adv_raw_std,
            'adv_raw_max': adv_raw_max,
            'adv_raw_min': adv_raw_min,
            'adv_norm_mean': adv_norm_mean,
            'adv_norm_std': adv_norm_std,
            'adv_norm_max': adv_norm_max,
            'adv_norm_min': adv_norm_min,
            'ratio_decision_mean': ratio_decision_mean_sum / self.policy_update_epochs,
            'ratio_decision_std': ratio_decision_std_sum / self.policy_update_epochs,
            'ratio_decision_p05': ratio_decision_p05_sum / self.policy_update_epochs,
            'ratio_decision_p95': ratio_decision_p95_sum / self.policy_update_epochs,
            'clipfrac_decision': clipfrac_decision_sum / self.policy_update_epochs,
            'approx_kl_decision': approx_kl_decision_sum / self.policy_update_epochs,
            'ratio_grab_mean': ratio_grab_mean_sum / self.policy_update_epochs,
            'clipfrac_grab': clipfrac_grab_sum / self.policy_update_epochs,
            'approx_kl_grab': approx_kl_grab_sum / self.policy_update_epochs,
            'ratio_step_mean': ratio_step_mean_sum / self.policy_update_epochs,
            'clipfrac_step': clipfrac_step_sum / self.policy_update_epochs,
            'approx_kl_step': approx_kl_step_sum / self.policy_update_epochs,
            'returns_mean': returns_mean,
            'returns_std': returns_std,
            'values_mean': values_mean_sum / self.policy_update_epochs,
            'values_std': values_std_sum / self.policy_update_epochs,
            'explained_variance': explained_variance_sum / self.policy_update_epochs,
        }

        # 清空缓冲区
        self._clear_buffer()

        return loss_discrete / self.policy_update_epochs,loss_continuous / self.policy_update_epochs