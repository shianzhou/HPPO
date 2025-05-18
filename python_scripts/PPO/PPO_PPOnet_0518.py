import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        super().__init__()
        self.node_num = node_num
        
        # 简化卷积网络
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        
        # 添加BatchNorm层
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 减小全连接层维度
        self.fc0 = nn.Linear(in_features=7200, out_features=512)  # 根据实际输出尺寸调整
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=4, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        
        # 简化图神经网络
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 128, 2, aggr='add')
        self.fc_graph = nn.Linear(128, 64)
        
        # 添加Dropout
        self.dropout = nn.Dropout(0.2)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=256, out_features=128)  # 128+64+64=256
        
        # Actor头：输出动作概率
        self.actor = nn.Linear(128, act_dim)
        
        # Critic头：输出状态值
        self.critic = nn.Linear(128, 1)
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
        # 图像处理
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        if len(x.shape) == 2:  # 如果是2D，添加批次和通道维度
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:  # 如果是3D，只添加通道维度
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)  # 展平
        if len(x.shape) == 1:  # 如果是1D，添加批次维度
            x = x.unsqueeze(0)
        
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        # 状态处理
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        if len(state.shape) == 1:  # 如果是1D，添加批次维度
            state = state.unsqueeze(0)
        
        state = self.fc2(state)
        state = self.relu(state)
        state = self.fc3(state)
        state = self.relu(state)
        
        # 图处理
        graph = self.creat_graph(x_graph)
        edge_index = graph.edge_index
        graph_feat = self.conv_graph1(graph.x, edge_index)
        graph_feat = self.relu(graph_feat)
        graph_feat = graph_feat.mean(dim=0).unsqueeze(0)  # 全局平均池化
        graph_feat = self.fc_graph(graph_feat)
        graph_feat = self.relu(graph_feat)
        
        # 特征融合
        combined = torch.cat([x, state, graph_feat], dim=1)
        features = self.fc4(combined)
        features = self.relu(features)
        features = self.dropout(features)
        
        # Actor输出
        action_probs = torch.softmax(self.actor(features), dim=1)
        
        # Critic输出
        value = self.critic(features)
        
        return action_probs, value

class PPO:
    def __init__(self, node_num, env_information):
        self.node_num = node_num
        self.env_information = env_information
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.value_coef = 0.5  # 值函数系数
        self.entropy_coef = 0.01  # 熵系数
        self.max_grad_norm = 0.5  # 梯度裁剪阈值
        
        # 学习率和优化器参数
        self.lr = 3e-4
        self.lr_decay = 0.995  # 学习率衰减
        
        # PPO更新参数
        self.update_epochs = 10  # 每批数据的更新次数
        self.batch_size = 64  # 批大小
        
        # 初始化策略网络
        self.policy = ActorCritic(act_dim=2, node_num=self.node_num).to(device)
        
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 存储轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 训练指标
        self.episode_rewards = []
        self.avg_rewards = []

    def choose_action(self, episode_num, obs, x_graph, explore=None):
        # 确保输入格式正确
        if isinstance(obs, list) and len(obs) >= 2:
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph  # 假设在这种情况下x_graph实际上是state
        
        # 使用递减的epsilon，从0.9开始逐渐降低到0.1
        epsilon = max(0.1, 0.90 - episode_num * 0.0001)  # 减小衰减速率，使探索率保持更高
        
        # 如果显式指定了explore参数，则使用该值
        if explore is not None:
            use_random = explore
        else:
            # 否则根据epsilon决定是否探索
            random_num = np.random.uniform()
            use_random = random_num < epsilon
        
        with torch.no_grad():
            # 获取动作概率和状态值
            action_probs, value = self.policy(x, state, x_graph)
            
            if use_random:
                # 探索：随机选择动作
                action = np.random.randint(0, action_probs.shape[1])
                action = torch.tensor(action, device=device)
            else:
                # 利用：使用策略网络生成动作
                # 创建分类分布
                dist = torch.distributions.Categorical(action_probs)
                # 采样动作
                action = dist.sample()
            
            # 计算动作的对数概率（即使是随机动作也要计算）
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)
        
        # 返回动作、对数概率和状态值
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        # 确保state和next_state格式正确
        if isinstance(state, list) and len(state) >= 3:
            self.states.append(state)  # 直接存储完整的state列表
        else:
            # 如果格式不匹配，尝试构建兼容格式
            self.states.append([state, state, state])
        
        self.actions.append(action)
        self.rewards.append(reward)
        
        if isinstance(next_state, list) and len(next_state) >= 3:
            self.next_states.append(next_state)
        else:
            self.next_states.append([next_state, next_state, next_state])
        
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def calculate_advantages(self):
        # 将rewards和values转换为numpy数组以便处理
        rewards = np.array(self.rewards)
        values = np.array(self.values)
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

    def learn(self):
     
        # 计算优势函数和回报
        advantages, returns = self.calculate_advantages()
        
        # 将数据转换为张量
        batch_states = self.states
        batch_actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        batch_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
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
                action_probs_batch = []
                values_batch = []
                
                for i in range(len(batch_x)):
                    action_probs, value = self.policy(batch_x[i], batch_state[i], batch_x_graph[i])
                    action_probs_batch.append(action_probs)
                    values_batch.append(value)
                
                action_probs_batch = torch.cat(action_probs_batch)
                values_batch = torch.cat(values_batch)
                
                # 获取当前批次的动作、对数概率等
                batch_actions_curr = batch_actions[batch_indices]
                batch_log_probs_curr = batch_log_probs[batch_indices]
                batch_advantages_curr = batch_advantages[batch_indices]
                batch_returns_curr = batch_returns[batch_indices]
                
                # 计算新的对数概率
                dist = torch.distributions.Categorical(action_probs_batch)
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
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return total_loss / self.update_epochs