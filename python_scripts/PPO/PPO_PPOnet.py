from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        super().__init__()
        self.node_num = node_num
        
        # 保留原有的特征提取网络结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 图神经网络部分
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        # --- 【核心修改 1】修改Actor头 ---
        # Actor不再输出一个离散概率，而是输出一个分布的参数
        # 1. mu_layer: 用于输出正态分布的均值(mu)
        # 2. log_sigma_layer: 用于输出log(sigma)，以保证sigma为正
        self.actor_mu = nn.Sequential(
            nn.Linear(200, act_dim),
            nn.Tanh()  # Tanh激活函数将mu的范围限制在[-1, 1]
        )
        
        # 将log_sigma作为可学习的参数，而不是依赖于状态。这是一种常见且稳定的做法。
        # act_dim 应该是动作的维度，这里是1
        self.actor_log_sigma = nn.Parameter(torch.zeros(act_dim))
        
        # Critic头：输出状态值
        self.critic = nn.Linear(200, 1)
    
    # 保留原有的图处理函数
    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        # 特征提取部分与原DQN相同
        self.graph = self.creat_graph(x_graph)
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, dim=0)
        #x = torch.unsqueeze(x, dim=1)
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
        #print(f"State shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
        #print(f"State: {state}")
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        
        x_graph = self.creat_graph(x_graph)
        edge_index = x_graph.edge_index
        x_graph = self.conv_graph1(x_graph.x, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph2(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph3(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph4(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph5(x_graph, edge_index)
        x_graph = torch.mean(x_graph, dim=0)
        x_graph = self.fc_graph(x_graph)
        
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # Actor: 输出均值 mu
        mu = self.actor_mu(features)
        
        # 计算标准差 sigma
        # 使用exp来保证sigma是正数。广播log_sigma以匹配mu的批次大小
        log_sigma = self.actor_log_sigma.expand_as(mu)
        sigma = torch.exp(log_sigma)
        
        # 构建正态分布
        dist = Normal(mu, sigma)
        
        # Critic: 输出状态值 (保持不变)
        value = self.critic(features)
        
        return dist, value

class PPO:
    def __init__(self, node_num, env_information):
        self.node_num = node_num
        self.env_information = env_information
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.value_coef = 0.5  # 值函数损失系数
        self.entropy_coef = 0.01  # 熵正则化系数
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

        # 学习率和优化器参数
        self.lr = 2e-4
        self.lr_decay = 0.995  # 学习率衰减

        # PPO更新参数
        self.update_epochs = 10  # 每批数据的更新次数
        self.batch_size = 64  # 批大小

        # 初始化策略网络
        self.policy = ActorCritic(act_dim=1, node_num=self.node_num).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 存储轨迹数据
        self.states = []
        self.action_shoulder = []
        self.action_arm = []
        self.rewards = []
        self.next_states = []
        self.values_shoulder = []
        self.values_arm = []   
        self.log_probs_shoulder = []
        self.log_probs_arm = []
        self.dones = []
    
    def choose_action(self, episode_num, obs, x_graph, action_type: str, explore=None):
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph


        # x = obs[0]
        # state = obs[1]

        # 确保输入张量在正确的设备上
        if isinstance(x, torch.Tensor):
            x = x.to(device)

        # 使用递减的epsilon，从0.9开始逐渐降低到0.1
        epsilon = max(0.1, 0.90 - episode_num * 0.0001)

        # 如果显式指定了explore参数，则使用该值
        if explore is not None:
            use_random = explore
        else:
            # 否则根据epsilon决定是否探索
            random_num = np.random.uniform()
            use_random = random_num < epsilon
        
        with torch.no_grad():
            # 策略网络输出动作分布的均值 mu 和价值 value
            if action_type == 'shoulder':
                #mu_shoulder, sigma_shoulder, value_shoulder = self.policy(x, state, x_graph)
                dist, value = self.policy(x, state, x_graph)
                mu_shoulder = dist.mean   #获取分布的均值
                sigma_shoulder = dist.stddev   #获取分布的标准差
                # 创建一个正态分布，用于计算对数概率和在利用时采样
                dist_shoulder = torch.distributions.Normal(mu_shoulder, sigma_shoulder)

                if use_random:
                    # 探索：在[-1, 1]范围内随机选择动作
                    act_dim = 1
                    action_scaled_shoulder = torch.tensor(np.random.uniform(-1, 1, size=act_dim), dtype=torch.float32).to(device)
                else:
                    # 利用：从策略网络生成的分布中采样
                    action_raw_shoulder = dist_shoulder.sample()
                    action_scaled_shoulder = torch.tanh(action_raw_shoulder)

                # 无论动作如何选择，都计算其在当前策略下的对数概率
                # 这是 on-policy 算法的要求
                action_raw_for_log_prob_shoulder = torch.atanh(torch.clamp(action_scaled_shoulder, -0.9999, 0.9999))
                log_prob_shoulder = dist_shoulder.log_prob(action_raw_for_log_prob_shoulder).sum(axis=-1)

                # 返回选择的动作、其对数概率和状态值
                return action_scaled_shoulder.cpu().numpy(), log_prob_shoulder.item(), value.item()
            elif action_type == 'arm':
                #mu_arm, sigma_arm, value_arm = self.policy(x, state, x_graph)
                dist, value = self.policy(x, state, x_graph)
                mu_arm = dist.mean
                sigma_arm = dist.stddev
                # 创建一个正态分布，用于计算对数概率和在利用时采样
                dist_arm = torch.distributions.Normal(mu_arm, sigma_arm)

                if use_random:
                    # 探索：在[-1, 1]范围内随机选择动作
                    act_dim = 1
                    action_scaled_arm = torch.tensor(np.random.uniform(-1, 1, size=act_dim), dtype=torch.float32).to(device)
                else:
                    # 利用：从策略网络生成的分布中采样
                    action_raw_arm = dist_arm.sample()
                    action_scaled_arm = torch.tanh(action_raw_arm)

                # 无论动作如何选择，都计算其在当前策略下的对数概率
                # 这是 on-policy 算法的要求
                action_raw_for_log_prob_arm = torch.atanh(torch.clamp(action_scaled_arm, -0.9999, 0.9999))
                log_prob_arm = dist_arm.log_prob(action_raw_for_log_prob_arm).sum(axis=-1)

                # 返回选择的动作、其对数概率和状态值
                return action_scaled_arm.cpu().numpy(), log_prob_arm.item(), value.item()

    
    def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value_shoulder, value_arm, log_prob_shoulder, log_prob_arm):
        """
        存储轨迹数据
        动作值范围：[-1.0, 1.0]
        """
        self.states.append(state)
        self.action_shoulder.append(action_shoulder)
        self.action_arm.append(action_arm)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values_shoulder.append(value_shoulder)
        self.values_arm.append(value_arm)
        self.log_probs_shoulder.append(log_prob_shoulder)
        self.log_probs_arm.append(log_prob_arm)
        self.dones.append(done)
    
    def calculate_advantages(self, action_type: str):
        """
        计算优势函数和回报
        
        :param action_type: 'shoulder' 或 'arm'，指定计算哪个动作的优势函数
        """
        if not self.rewards:          # 没有数据直接返回空
            return np.array([]), np.array([])

        if action_type == 'shoulder':
            values = np.array(self.values_shoulder)
        elif action_type == 'arm':
            values = np.array(self.values_arm)
        else:
            raise ValueError("action_type 必须是 'shoulder' 或 'arm'")
        
        # 将rewards和values转换为numpy数组以便处理
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # 计算GAE优势函数
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        # 从后向前计算优势函数
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 对于最后一个时间步，使用0作为下一个值的估计
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # 计算回报
        returns = advantages + values

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self, action_type: str):
        """
        根据指定的动作类型（'shoulder' 或 'arm'）更新策略网络。

        :param action_type: 一个字符串，'shoulder' 或 'arm'，用于指定要更新哪个动作部分。
        """
        # 检查传入的参数是否合法
        if action_type not in ['shoulder', 'arm']:
            raise ValueError("action_type 必须是 'shoulder' 或 'arm'")
        
        # 计算优势函数和回报
        advantages, returns = self.calculate_advantages(action_type)
        if len(advantages) == 0:
            return 0.0

        # 将数据转换为张量
        batch_states = self.states
        batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
         # 根据 action_type 选择相应的动作和对数概率
        if action_type == 'shoulder':
            batch_actions = torch.tensor(self.action_shoulder, dtype=torch.float32).to(device)
            batch_log_probs = torch.tensor(self.log_probs_shoulder, dtype=torch.float32).to(device)
        elif action_type == 'arm': 
            batch_actions = torch.tensor(self.action_arm, dtype=torch.float32).to(device)
            batch_log_probs = torch.tensor(self.log_probs_arm, dtype=torch.float32).to(device)
        # 多次更新网络
        total_loss = 0
        for _ in range(self.update_epochs):
            # 生成随机索引
            indices = torch.randperm(len(batch_states))

            # 分批处理数据
            for start_idx in range(0, len(batch_states), self.batch_size):
                # 获取当前批次的索引
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # 处理当前批次的状态
                batch_x = []
                batch_state = []
                batch_x_graph = []

                for idx in batch_indices:
                    if idx < len(batch_states):
                        batch_x.append(batch_states[idx][0])
                        batch_state.append(batch_states[idx][1])
                        batch_x_graph.append(batch_states[idx][2])

                # 如果批次为空，跳过
                if not batch_x:
                    continue

                # 前向传播
                mu_batch_shoulder_list = []
                sigma_batch_shoulder_list = []
                mu_batch_arm_list = []
                sigma_batch_arm_list = []
                values_batch = []
                
                if action_type == 'shoulder':
                    for i in range(len(batch_x)):
                        #(mu_shoulder, sigma_shoulder), value = self.policy(batch_x[i], batch_state[i], batch_x_graph[i])
                        dist, value = self.policy(batch_x[i], batch_state[i], batch_x_graph[i])
                        mu_shoulder = dist.mean   #获取分布的均值
                        sigma_shoulder = dist.stddev   #获取分布的标准差                       
                        mu_batch_shoulder_list.append(mu_shoulder)
                        sigma_batch_shoulder_list.append(sigma_shoulder)
                        values_batch.append(value)
                else: # action_type == 'arm'
                    for i in range(len(batch_x)):
                        dist, value = self.policy(batch_x[i], batch_state[i], batch_x_graph[i])
                        mu_arm = dist.mean   #获取分布的均值
                        sigma_arm = dist.stddev   #获取分布的标准差          
                        mu_batch_arm_list.append(mu_arm)
                        sigma_batch_arm_list.append(sigma_arm)
                        values_batch.append(value)
                values_batch = torch.cat(values_batch)

                # --- 根据 action_type 计算策略损失和熵 ---
                if action_type == 'shoulder':
                    mu_batch = torch.cat(mu_batch_shoulder_list)
                    sigma_batch = torch.cat(sigma_batch_shoulder_list)
                else: # action_type == 'arm'
                    mu_batch = torch.cat(mu_batch_arm_list)
                    sigma_batch = torch.cat(sigma_batch_arm_list)

                # 获取当前批次的动作、对数概率等
                batch_actions_curr = batch_actions[batch_indices]
                batch_log_probs_curr = batch_log_probs[batch_indices]
                batch_advantages_curr = batch_advantages[batch_indices]
                batch_returns_curr = batch_returns[batch_indices]

                # 计算新的对数概率
                dist = torch.distributions.Normal(mu_batch, sigma_batch)
                new_log_probs = dist.log_prob(batch_actions_curr)
                entropy = dist.entropy().mean()

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_log_probs_curr)
                # PPO裁剪目标
                surr1 = ratio * batch_advantages_curr
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages_curr
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失
                value_loss = nn.MSELoss()(values_batch, batch_returns_curr)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                total_loss += loss.item()

        # 更新学习率
        self.scheduler.step()

        # 清空轨迹数据
        self.states = []
        self.action_shoulder = []
        self.action_arm = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        # 重置每回合对应的价值估计，防止累积导致长度不一致
        self.values_shoulder = []
        self.values_arm = []
        self.log_probs_shoulder = []
        self.log_probs_arm = []
        self.x_graphs = []

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("total_loss:", total_loss)
        return total_loss / self.update_epochs