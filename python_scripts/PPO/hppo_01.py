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
        # 图像特征
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        # 上游传入的x通常已是[C,H,W]=(1,H,W)，与PPO保持一致，这里仅增加batch维
        x = torch.unsqueeze(x, dim=0)  # [N=1,C,H,W]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))
        # 状态特征
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        # 图特征（简化为全连接）
        x_graph = torch.as_tensor(x_graph, dtype=torch.float32).to(device)
        x_graph = self.fc_graph(x_graph)
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        # 融合

        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)


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

            discrete_action = torch.cat((decision.view(1), grab_discrete, step_discrete), dim=0)
            continuous_action = torch.cat((grab_continuous, step_continuous), dim=0)

            discrete_log_prob = torch.cat(
                (decision_log_prob.view(1), grab_discrete_log_prob, step_discrete_log_prob),
                dim=0
            )
            continuous_log_prob = torch.cat(
                (
                    grab_cont_dist.log_prob(grab_continuous),
                    step_cont_dist.log_prob(step_continuous),
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
            return 0,0

        # 计算优势函数和回报（当前实现正确，无需修改）
        advantages = self.calculate_advantages()
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

        # 转换为张量并移动到设备
        batch_states = self.states  # 列表，每个元素是状态元组
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_continuous = 0
        loss_discrete = 0

        for _ in range(self.policy_update_epochs):
            all_decision_dists = []
            all_grab_dists = []
            all_step_dists = []
            all_grab_cont_dists = []
            all_step_cont_dists = []
            all_values = []

            # 批量处理状态
            for i in range(len(batch_states)):
                decision_logits, grab_logits, step_logits, grab_mu, step_mu, value = self.policy(
                    x=batch_states[i][0],
                    state=batch_states[i][1],
                    x_graph=batch_states[i][2]
                )
                decision_dist, grab_dist, step_dist, grab_cont_dist, step_cont_dist = self._build_distributions(
                    decision_logits, grab_logits, step_logits, grab_mu, step_mu
                )

                all_decision_dists.append(decision_dist)
                all_grab_dists.append(grab_dist)
                all_step_dists.append(step_dist)
                all_grab_cont_dists.append(grab_cont_dist)
                all_step_cont_dists.append(step_cont_dist)
                all_values.append(value.squeeze(0))

            new_decision_log_probs = torch.stack(
                [all_decision_dists[i].log_prob(batch_decisions[i]) for i in range(len(all_decision_dists))]
            )
            new_grab_discrete_log_probs = torch.stack(
                [all_grab_dists[i].log_prob(batch_grab_discrete_actions[i]).squeeze(0) for i in range(len(all_grab_dists))]
            )
            new_step_discrete_log_probs = torch.stack(
                [all_step_dists[i].log_prob(batch_step_discrete_actions[i]).squeeze(0) for i in range(len(all_step_dists))]
            )
            new_grab_cont_log_probs = torch.stack(
                [all_grab_cont_dists[i].log_prob(batch_grab_continuous_actions[i]).sum(-1).squeeze(0)
                 for i in range(len(all_grab_cont_dists))]
            )
            new_step_cont_log_probs = torch.stack(
                [all_step_cont_dists[i].log_prob(batch_step_continuous_actions[i]).sum(-1).squeeze(0)
                 for i in range(len(all_step_cont_dists))]
            )

            all_values = torch.stack(all_values)

            decision_mask = torch.ones_like(advantages)
            grab_mask = (batch_decisions == 0).float()
            step_mask = ((batch_decisions == 1).float() * batch_grab_success)

            old_grab_discrete_log_probs = batch_grab_discrete_log_probs.sum(-1)
            old_step_discrete_log_probs = batch_step_discrete_log_probs.sum(-1)

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

        # 清空缓冲区
        self._clear_buffer()

        return loss_discrete / self.policy_update_epochs,loss_continuous / self.policy_update_epochs