import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import torch_geometric
from controller import Robot
from torch_geometric.data import Data

from python_scripts.Project_config import device


class HPPOActorCritic(nn.Module):
    """
    Actor-Critic 网络，输出多维离散动作（0/1，是否运动）。
    """

    def __init__(self, act_dim: int, node_num: int):
        super().__init__()
        self.node_num = node_num
        self.temperature = 1.0  # 控制离散动作探索程度

        # 图像特征提取
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=1)

        self.fc0 = nn.Linear(6272, 6000)
        self.fc1 = nn.Linear(6000, 100)

        # 状态特征
        self.fc2 = nn.Linear(20, 100)
        self.fc3 = nn.Linear(100, 100)

        # 图结构
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr="add")
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr="add")
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr="add")
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr="add")
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr="add")
        self.fc_graph = nn.Linear(1000, 100)

        self.fc4 = nn.Linear(300, 200)

        # 离散动作 logits（直接输出 logits，便于构造 Bernoulli 分布）
        self.actor_logits = nn.Linear(200, act_dim)
        self.critic = nn.Linear(200, 1)

    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)

    def creat_x(self, x_graph):
        ans = [[] for _ in range(self.node_num)]
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

    def _safe_normalize(self, tensor):
        # 检查并处理NaN和Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        denom = max_val - min_val
        if torch.abs(denom) < 1e-6:
            return torch.zeros_like(tensor)
        normalized = torch.div(torch.sub(tensor, min_val), denom)
        # 再次检查归一化后的结果
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized

    def forward(self, x, state, x_graph):
        graph = self.creat_graph(x_graph)
        edge_index = graph.edge_index

        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        # 检查输入中的NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self._safe_normalize(x)

        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        # 检查状态中的NaN/Inf
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = self.fc2(state)
        state = self.fc3(state)
        state = self._safe_normalize(state)

        x_graph = self.conv_graph1(graph.x, edge_index)
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
        x_graph = self._safe_normalize(x_graph)

        state_x = torch.cat((x, state, x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # 检查features中的NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        logits = self.actor_logits(features)
        # 温度越高越接近随机，越低越偏向确定性
        temp = max(self.temperature, 1e-3)
        logits = logits / temp
        value = self.critic(features)
        
        # 检查输出中的NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.isnan(value).any() or torch.isinf(value).any():
            value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return logits, value

    def set_temperature(self, temperature: float):
        self.temperature = max(temperature, 1e-2)


class HPPOAgent:
    """
    层级 PPO：负责输出 0/1 离散动作，用于决定舵机是否运动。
    """

    def __init__(self, node_num: int, action_splits: dict):
        """
        Args:
            node_num: 图节点数
            action_splits: dict, 例如 {'upper': 2, 'lower': 3}
        """
        self.node_num = node_num
        self.total_act_dim = sum(action_splits.values())
        self.action_splits = action_splits
        self.group_slices = {}
        start = 0
        for key, span in action_splits.items():
            self.group_slices[key] = slice(start, start + span)
            start += span

        self.policy = HPPOActorCritic(self.total_act_dim, node_num).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.1
        self.value_coef = 0.5
        self.entropy_coef = 0.001
        self.max_grad_norm = 0.5  # 降低梯度裁剪阈值，防止梯度爆炸
        self.update_epochs = 4
        self.batch_size = 64

        self.buffers = {
            key: {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "log_probs": [],
                "values": [],
            }
            for key in action_splits.keys()
        }

    def _get_buffer(self, group):
        if group not in self.buffers:
            raise ValueError(f"未知的 group: {group}")
        return self.buffers[group]

    def choose_action(self, group, obs, x_graph, episode_num=None):
        """
        返回指定 group 的离散动作（0/1）
        """
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph

        temperature = self._compute_temperature(episode_num or 0)
        self.policy.set_temperature(temperature)

        logits, value = self.policy(x, state, x_graph)
        idx = self.group_slices[group]
        group_logits = logits[idx]
        
        # 检查并处理NaN值
        if torch.isnan(group_logits).any() or torch.isinf(group_logits).any():
            print(f"警告: 在choose_action中检测到NaN/Inf logits，使用零值替换")
            group_logits = torch.nan_to_num(group_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 限制logits的范围，避免数值不稳定
        group_logits = torch.clamp(group_logits, min=-10.0, max=10.0)

        dist = Bernoulli(logits=group_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return (
            action.detach().cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def _compute_temperature(self, episode_num):
        start_temp = 1.5
        end_temp = 0.4
        decay_episode = 4000
        progress = min(max(episode_num, 0) / decay_episode, 1.0)
        return start_temp + (end_temp - start_temp) * progress

    def store_transition(self, group, state, action, reward, next_state, done, value, log_prob):
        buf = self._get_buffer(group)
        buf["states"].append(state)
        buf["actions"].append(action)
        buf["rewards"].append(reward)
        buf["next_states"].append(next_state)
        buf["dones"].append(done)
        buf["log_probs"].append(log_prob)
        buf["values"].append(value)

    def _calculate_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_value = values[i + 1] if i < len(rewards) - 1 else 0
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=device)
        
        # 检查并处理NaN/Inf
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 标准化优势值
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()
        
        # 再次检查标准化后的结果
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return advantages, returns

    def learn(self, group):
        buf = self._get_buffer(group)
        if len(buf["states"]) == 0:
            return 0.0

        rewards = torch.tensor(buf["rewards"], dtype=torch.float32, device=device)
        values = torch.tensor(buf["values"], dtype=torch.float32, device=device)
        dones = torch.tensor(buf["dones"], dtype=torch.float32, device=device)
        advantages, returns = self._calculate_advantages(rewards.tolist(), values.tolist(), dones.tolist())

        old_log_probs = torch.tensor(buf["log_probs"], dtype=torch.float32, device=device)
        actions = torch.tensor(buf["actions"], dtype=torch.float32, device=device)

        total_loss = 0.0
        data_size = len(buf["states"])

        for _ in range(self.update_epochs):
            indices = torch.randperm(data_size)
            for start in range(0, data_size, self.batch_size):
                idx_batch = indices[start : start + self.batch_size]
                if len(idx_batch) == 0:
                    continue

                batch_states = [buf["states"][i] for i in idx_batch]
                batch_actions = actions[idx_batch]
                batch_advantages = advantages[idx_batch]
                batch_returns = returns[idx_batch]
                batch_old_log_probs = old_log_probs[idx_batch]

                new_log_probs_list = []
                entropy_list = []
                values_list = []

                for sample_state, sample_action in zip(batch_states, batch_actions):
                    logits, sample_value = self.policy(
                        sample_state[0], sample_state[1], sample_state[2]
                    )
                    sample_logits = logits[self.group_slices[group]]
                    
                    # 检查并处理NaN值
                    if torch.isnan(sample_logits).any() or torch.isinf(sample_logits).any():
                        print(f"警告: 检测到NaN/Inf logits，使用零值替换")
                        sample_logits = torch.nan_to_num(sample_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # 限制logits的范围，避免数值不稳定
                    sample_logits = torch.clamp(sample_logits, min=-10.0, max=10.0)
                    
                    dist = Bernoulli(logits=sample_logits)
                    values_list.append(sample_value)

                    action_tensor = sample_action
                    log_prob = dist.log_prob(action_tensor).sum()
                    new_log_probs_list.append(log_prob)
                    entropy_list.append(dist.entropy().mean())

                new_log_probs = torch.stack(new_log_probs_list)
                entropy = torch.stack(entropy_list).mean()
                values_tensor = torch.stack(values_list).squeeze()
                
                # 检查计算过程中的NaN/Inf
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    print(f"警告: new_log_probs包含NaN/Inf，跳过此批次")
                    continue
                if torch.isnan(batch_old_log_probs).any() or torch.isinf(batch_old_log_probs).any():
                    print(f"警告: batch_old_log_probs包含NaN/Inf，跳过此批次")
                    continue

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # 限制ratio的范围，防止数值爆炸
                ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_tensor, batch_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 检查loss中的NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: loss包含NaN/Inf，跳过此批次")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        self.scheduler.step()

        # 清空缓冲
        for key in buf.keys():
            buf[key] = []
        self.buffers[group] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }

        return total_loss / max(1, (data_size / self.batch_size) * self.update_epochs)

