import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Bernoulli
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
        self.fc2 = nn.Linear(20, 100)
        self.fc3 = nn.Linear(100, 100)
        # 图神经网络部分（可选，简化版）
        self.fc_graph = nn.Linear(20, 100)
        # 共享特征层
        self.fc4 = nn.Linear(300, 200)
        # 多舵机离散动作头
        self.discrete_head = nn.Linear(200, num_servos)
        # Critic头
        self.critic = nn.Linear(200, 1)
        
        # 添加权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier初始化避免梯度爆炸/消失"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, state, x_graph):
        # 图像特征
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        
        # 添加NaN检查
        if torch.isnan(x).any():
            print("⚠️ 输入x包含NaN")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = torch.unsqueeze(x, dim=0)  # [1,C,H,W]
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
        range1 = max_val1 - min_val1
        # 如果全是常数，直接设为0
        if range1 < 1e-8:
            normalized_data1 = torch.zeros_like(x)
        else:
            normalized_data1 = (x - min_val1) / (range1 + 1e-8)
        
        # 状态特征
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        if torch.isnan(state).any():
            print("⚠️ 输入state包含NaN")
            state = torch.nan_to_num(state, nan=0.0)
            
        state = self.fc2(state)
        state = self.fc3(state)
        
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        range2 = max_val2 - min_val2
        if range2 < 1e-8:
            normalized_data2 = torch.zeros_like(state)
        else:
            normalized_data2 = (state - min_val2) / (range2 + 1e-8)
        
        # 图特征
        x_graph = torch.as_tensor(x_graph, dtype=torch.float32).to(device)
        if torch.isnan(x_graph).any():
            print("⚠️ 输入x_graph包含NaN")
            x_graph = torch.nan_to_num(x_graph, nan=0.0)
            
        x_graph = self.fc_graph(x_graph)
        
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        range3 = max_val3 - min_val3
        if range3 < 1e-8:
            normalized_x_graph = torch.zeros_like(x_graph)
        else:
            normalized_x_graph = (x_graph - min_val3) / (range3 + 1e-8)
        
        # 融合
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # 多舵机离散动作概率
        discrete_logits = self.discrete_head(features)
        discrete_probs = torch.sigmoid(discrete_logits)  # [num_servos]
        
        # 最终安全检查
        if torch.isnan(discrete_probs).any():
            print("⚠️ discrete_probs包含NaN，强制设为0.5")
            discrete_probs = torch.full_like(discrete_probs, 0.5)
        
        value = self.critic(features)
        return discrete_probs, value

class HPPO:
    def __init__(self, num_servos, node_num, env_information=None ):
        self.num_servos = num_servos
        self.node_num = node_num
        self.env_information = env_information
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.90  # 🔧 降低从 0.95 -> 0.90，减缓 GAE 增长速度
        self.clip_ratio = 0.15  # 🔧 降低从 0.2 -> 0.15，更保守的策略更新
        self.policy_update_epochs = 2  # 🔧 降低从 3 -> 2，减少每个 batch 的更新轮数
        self.value_coef = 0.25  # 🔧 降低从 0.5 -> 0.25，减弱价值损失权重
        self.entropy_coef = 0.01
        
        # 学习率设置 - 与其他PPO保持一致
        self.lr = 3e-5  # 🔧 进一步降低从 5e-5 -> 3e-5，更稳定的参数更新
        self.lr_decay = 0.995  # 学习率衰减
        
        # 网络
        self.policy = MultiDiscreteActorCritic(num_servos, node_num).to(device)
        
        # 优化器 - 使用更保守的学习率
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学习率调度器 - 添加学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        # 轨迹缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def choose_action(self, obs, x_graph):
        with torch.no_grad():
            discrete_probs, value = self.policy(x=obs[0], state=obs[1], x_graph=x_graph)
            m = Bernoulli(discrete_probs)
            discrete_actions = m.sample()  # [num_servos]
            discrete_log_probs = m.log_prob(discrete_actions)  # [num_servos]
            action = discrete_actions.cpu().numpy()
            log_prob = discrete_log_probs.cpu().numpy()
            value = value.item()
            # 修改3：添加返回值字典
            return {
                'discrete_action': action,
                'discrete_log_prob': log_prob,
                'value': value
            }

    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def calculate_advantages(self):
        advantages = []
        gae = 0
        max_gae = 10.0  # 🔧 添加 GAE 上界防止梯度爆炸
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            
            # 🔧 裁剪 GAE 防止指数级增长导致的梯度爆炸
            gae = np.clip(gae, -max_gae, max_gae)
            
            advantages.insert(0, gae)
        
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # 标准化处理
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return advantages_tensor

    def learn(self):
        if len(self.states) < 32:
            return 0

        advantages = self.calculate_advantages()
        
        # 🔧 异常值检测和保护
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("⚠️ 警告：Advantage 包含 NaN 或 Inf，跳过此次学习")
            # 清空缓冲区
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.values = []
            self.log_probs = []
            self.dones = []
            return 0
        
        max_advantage = torch.abs(advantages).max().item()
        if max_advantage > 50:
            print(f"⚠️ 警告：Advantage 过大 ({max_advantage:.2f})，进行强制裁剪")
            advantages = torch.clamp(advantages, -10, 10)
        
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
        batch_states = self.states
        batch_discrete_actions = torch.tensor(self.actions, dtype=torch.float32).to(device)  # shape: [batch, num_servos]
        batch_discrete_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        total_loss = 0

        for _ in range(self.policy_update_epochs):
            all_discrete_probs = []
            all_values = []
            for i in range(len(batch_states)):
                discrete_probs, value = self.policy(x=batch_states[i][0], state=batch_states[i][1], x_graph=batch_states[i][2])
                all_discrete_probs.append(discrete_probs)
                all_values.append(value)
            all_discrete_probs = torch.stack(all_discrete_probs)  # [batch, num_servos]
            all_values = torch.cat(all_values)
            # 离散部分
            m = Bernoulli(all_discrete_probs)
            new_discrete_log_probs = m.log_prob(batch_discrete_actions)
            discrete_ratio = torch.exp(new_discrete_log_probs - batch_discrete_log_probs)
            # 总ratio（离散部分取均值）
            total_ratio = discrete_ratio.mean(dim=1)

            surr1 = total_ratio * advantages
            surr2 = torch.clamp(total_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(all_values, returns)
            entropy = m.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            
            # 更严格的梯度裁剪 - 与PPO2保持一致
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            total_loss += loss.item()

        # 更新学习率
        self.scheduler.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []

        return total_loss / self.policy_update_epochs