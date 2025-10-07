import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.distributions import Bernoulli
from torch.distributions import Normal
from torch.distributions import Categorical
from python_scripts.Project_config import device

class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, num_servos, node_num):
        super().__init__()
        self.num_servos = num_servos
        self.node_num = node_num
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
        # 多舵机离散动作头
        self.discrete_head = nn.Linear(200, num_servos)  # 输出num_servos个logit

        #连续动作头，输出值是已知所有动作的参数(输出维度因为所有需要参数的二倍)
        self.continuous = nn.Linear(200,num_servos*2)

        #且由于将sigma作为连续网络的输出值，所以不需要
        #self.actor_log_sigma = nn.Parameter(torch.zeros(act_dim))

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
        # 多舵机离散动作概率
        discrete_logits = self.discrete_head(features)
        #
        # discrete_probs = torch.sigmoid(discrete_logits)  # [num_servos]
        discrete_logits = discrete_logits.view(-1, self.num_servos, 2)  # 重塑为 [batch_size, num_servos, 3]
        discrete_probs = F.softmax(discrete_logits, dim=-1)  # 在最后一个维度（动作维度）应用 softmax
        discrete_dist = Categorical(logits=discrete_probs)
        #将连续层的输出拆分为均值和方差
        continuous_output = self.continuous(features)
        mu, log_sigma = torch.chunk(continuous_output, 2, dim=-1)  # 拆分
        sigma = torch.exp(log_sigma)

        # 构建正态分布
        continuous_dist = Normal(mu, sigma)

        value = self.critic(features)

        return discrete_dist,continuous_dist,value

class HPPO:
    def __init__(self, num_servos, node_num, env_information=None ):
        self.num_servos = num_servos
        self.node_num = node_num
        self.env_information = env_information
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_update_epochs = 10
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # 学习率设置 - 与其他PPO保持一致
        self.lr = 2e-4  # 降低学习率，与PPO保持一致
        self.lr_decay = 0.995  # 学习率衰减
        
        # 网络
        self.policy = MultiDiscreteActorCritic(num_servos, node_num).to(device)
        
        # 优化器 - 使用更保守的学习率
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学习率调度器 - 添加学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        # 轨迹缓存
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.dones = []

    def choose_action(self, episode_num, obs, x_graph):
        with torch.no_grad():
            discrete_dist, continuous_dist , value= self.policy(
                x=obs[0],
                state=obs[1],
                x_graph=x_graph
            )
            # discrete_probs, value = self.policy(x=obs[0], state=obs[1], x_graph=x_graph)
            # m = Bernoulli(discrete_probs)


            # discrete_actions = m.sample()  # [num_servos]
            # discrete_log_probs = m.log_prob(discrete_actions)  # [num_servos]
            # action = discrete_actions.cpu().numpy()
            # log_prob = discrete_log_probs.cpu().numpy()

            discrete_action = discrete_dist.sample()  # 采样离散动作
            continuous_action = continuous_dist.sample()  # 采样连续参数

            # 2. 计算已采样动作的对数概率
            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            continuous_log_prob = continuous_dist.log_prob(continuous_action)

            value = value.item()
            action_dict = {
                'discrete_action': discrete_action.cpu().numpy(),
                'continuous_action': continuous_action.cpu().numpy(),
                'discrete_log_prob': discrete_log_prob.cpu().numpy(),
                'continuous_log_prob': continuous_log_prob.cpu().numpy(),
                'value': value.item()  # 状态价值是标量
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
                         discrete_log_prob, continuous_log_prob):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.discrete_log_probs.append(discrete_log_prob)  # 存储离散对数概率
        self.continuous_log_probs.append(continuous_log_prob)  # 存储连续对数概率
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

    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().data.numpy().squeeze(0)

    def _clear_buffer(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.dones = []

    # def learn(self):
    #     if len(self.states) < 32:
    #         return 0
    #
    #
    #     advantages = self.calculate_advantages()
    #     returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
    #     batch_states = self.states
    #     batch_discrete_actions = torch.tensor(self.actions, dtype=torch.float32).to(device)  # shape: [batch, num_servos]
    #     batch_discrete_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #
    #
    #     total_loss = 0
    #
    #     for _ in range(self.policy_update_epochs):
    #         all_discrete_probs = []
    #         all_values = []
    #         for i in range(len(batch_states)):
    #             discrete_probs, value = self.policy(x=batch_states[i][0], state=batch_states[i][1], x_graph=batch_states[i][2])
    #             all_discrete_probs.append(discrete_probs)
    #             all_values.append(value)
    #         all_discrete_probs = torch.stack(all_discrete_probs)  # [batch, num_servos]
    #         all_values = torch.cat(all_values)
    #         # 离散部分
    #         m = Bernoulli(all_discrete_probs)
    #         new_discrete_log_probs = m.log_prob(batch_discrete_actions)
    #         discrete_ratio = torch.exp(new_discrete_log_probs - batch_discrete_log_probs)
    #         # 总ratio（离散部分取均值）
    #         total_ratio = discrete_ratio.mean(dim=1)
    #
    #         surr1 = total_ratio * advantages
    #         surr2 = torch.clamp(total_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
    #
    #         policy_loss = -torch.min(surr1, surr2).mean()
    #         value_loss = nn.MSELoss()(all_values, returns)
    #         entropy = m.entropy().mean()
    #
    #         loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #
    #         # 更严格的梯度裁剪 - 与PPO2保持一致
    #         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.1)
    #
    #         self.optimizer.step()
    #         total_loss += loss.item()
    #
    #     # 更新学习率
    #     self.scheduler.step()
    #
    #     self.states = []
    #     self.actions = []
    #     self.rewards = []
    #     self.next_states = []
    #     self.values = []
    #     self.log_probs = []
    #     self.dones = []
    #     return total_loss / self.policy_update_epochs
    def learn(self):
        if len(self.states) < self.batch_size:  # 使用定义的batch_size
            return 0

        # 计算优势函数和回报（当前实现正确，无需修改）
        advantages = self.calculate_advantages()
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

        # 转换为张量并移动到设备
        batch_states = self.states  # 列表，每个元素是状态元组
        batch_discrete_actions = torch.tensor(self.discrete_actions, dtype=torch.float32).to(self.device)
        batch_continuous_actions = torch.tensor(self.continuous_actions, dtype=torch.float32).to(self.device)
        batch_discrete_log_probs = torch.tensor(self.discrete_log_probs, dtype=torch.float32).to(self.device)
        batch_continuous_log_probs = torch.tensor(self.continuous_log_probs, dtype=torch.float32).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0

        for _ in range(self.policy_update_epochs):
            all_discrete_dists = []
            all_continuous_dists = []
            all_values = []

            # 批量处理状态
            for i in range(len(batch_states)):
                discrete_dist, continuous_dist, value = self.policy(
                    x=batch_states[i][0],
                    state=batch_states[i][1],
                    x_graph=batch_states[i][2]
                )
                all_discrete_dists.append(discrete_dist)
                all_continuous_dists.append(continuous_dist)
                all_values.append(value)

            # 计算新策略的对数概率
            new_discrete_log_probs = torch.stack(
                [dist.log_prob(batch_discrete_actions[i]) for i, dist in enumerate(all_discrete_dists)])
            new_continuous_log_probs = torch.stack(
                [dist.log_prob(batch_continuous_actions[i]) for i, dist in enumerate(all_continuous_dists)])
            all_values = torch.cat(all_values)

            # 计算总概率比（离散和连续对数概率之和）
            old_total_log_probs = batch_discrete_log_probs + batch_continuous_log_probs
            new_total_log_probs = new_discrete_log_probs + new_continuous_log_probs
            ratios = torch.exp(new_total_log_probs - old_total_log_probs)

            # PPO裁剪损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = nn.MSELoss()(all_values, returns)

            # 熵奖励（鼓励探索）
            discrete_entropy = torch.stack([dist.entropy().mean() for dist in all_discrete_dists]).mean()
            continuous_entropy = torch.stack([dist.entropy().mean() for dist in all_continuous_dists]).mean()
            entropy_bonus = discrete_entropy + continuous_entropy

            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()

        # 清空缓冲区
        self._clear_buffer()

        return total_loss / self.policy_update_epochs