import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.00001                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.99                                     # reward discount
TARGET_REPLACE_ITER = 100                        # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 100000


class Net(nn.Module):
    def __init__(self, act_dim, node_num):    # 初始化网络
        super().__init__()  
        self.node_num = node_num  # 点总数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)  # 卷积层1
        self.relu = nn.ReLU()  # 激活函数
        self.Sigmoid = nn.Sigmoid()  # 激活函数
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 池化层1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))  # 卷积层2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)  # 卷积层3    
        # 调整全连接层的维度
        self.fc0 = nn.Linear(in_features=6272, out_features=1024)  # 全连接层0  
        self.fc1 = nn.Linear(in_features=1024, out_features=256)  # 全连接层1
        self.fc2 = nn.Linear(in_features=4, out_features=64)  # 全连接层2
        self.fc3 = nn.Linear(in_features=64, out_features=64)  # 全连接层3
        
        # 调整融合特征的维度
        self.fc_fusion = nn.Linear(in_features=256+64+100, out_features=256)  # 融合层
        
        # 状态值流和优势流
        self.fc_value = nn.Linear(in_features=256, out_features=1)  # 状态值流
        self.fc_advantage = nn.Linear(in_features=256, out_features=1)  # 优势流
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')    # 图卷积层1
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')  # 图卷积层2
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')  # 图卷积层3
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')  # 图卷积层4
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')  # 图卷积层5
        self.fc_graph = nn.Linear(1000, 100)  # 全连接层6


    def create_edge_index(self):    # 创建图的边索引
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],

            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    # 为图中的节点生成信息
    def creat_x(self, x_graph):
        # 确保输入的图数据长度与网络定义的节点数匹配
        if len(x_graph) != self.node_num:
            raise ValueError(f"输入的图数据长度({len(x_graph)})与网络定义的节点数({self.node_num})不匹配")
        
        ans = [[] for _ in range(self.node_num)]    # 创建图的节点信息
        for i in range(self.node_num):
            ans[i] = [x_graph[i]]
        return ans
    
    # 结合前面的两个函数生成图神经网络使用的图
    def creat_graph(self, x_graph):
        # 创建图的节点特征
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        
        # 创建图的边索引
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        
        # 创建图数据
        graph = Data(x=x, edge_index=edge_index)
        
        # 将图数据移动到GPU
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        
        return graph

    def forward(self, x, state, x_graph, action=None):
        """
        前向传播
        :param x: 图像输入
        :param state: 状态输入
        :param x_graph: 图结构输入
        :param action: 动作输入，如果为None则只返回状态值，否则返回对应的Q值
        :return: 如果action为None，返回状态值；否则返回对应的Q值
        """
        self.graph = self.creat_graph(x_graph)  # 创建图
        x = torch.tensor(x).to(device)  # 将输入数据转换为张量并移动到GPU
        x = torch.unsqueeze(x, dim=0)  # 在第0维增加一个维度，变成 [1, 128, 128]
        x = torch.unsqueeze(x, dim=1)  # 在第1维增加一个维度，变成 [1, 1, 128, 128]
        x = x.float()  # 将张量转换为浮点数
        
        # 图像特征提取
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)  # [256]
        x = self.relu(x)
        
        # 状态特征提取
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        else:
            state_tensor = state.to(device).float()
            
        state_features = self.fc2(state_tensor)
        state_features = self.relu(state_features)
        state_features = self.fc3(state_features)  # [64]
        state_features = self.relu(state_features)
        
        # 图特征提取
        x_graph_tensor = self.creat_graph(x_graph)
        edge_index = x_graph_tensor.edge_index
        x_graph_features = self.conv_graph1(x_graph_tensor.x, edge_index)
        x_graph_features = self.relu(x_graph_features)
        x_graph_features = self.conv_graph2(x_graph_features, edge_index)
        x_graph_features = self.relu(x_graph_features)
        x_graph_features = self.conv_graph3(x_graph_features, edge_index)
        x_graph_features = self.relu(x_graph_features)
        x_graph_features = self.conv_graph4(x_graph_features, edge_index)
        x_graph_features = self.relu(x_graph_features)
        x_graph_features = self.conv_graph5(x_graph_features, edge_index)
        x_graph_features = torch.mean(x_graph_features, dim=0)  # 全局平均池化
        x_graph_features = self.fc_graph(x_graph_features)  # [100]
        x_graph_features = self.relu(x_graph_features)
        
        # 特征融合
        fused_features = torch.cat([x, state_features, x_graph_features], dim=-1)
        
        # 确保特征维度正确
        if len(fused_features.shape) == 1:
            fused_features = fused_features.unsqueeze(0)  # 添加batch维度
            
        # 通过融合层
        fused_features = self.fc_fusion(fused_features)
        fused_features = self.relu(fused_features)
        
        # 状态值流和优势流
        state_value = self.fc_value(fused_features)  # [batch_size, 1]
        advantage = self.fc_advantage(fused_features)  # [batch_size, 1]
        
        if action is not None:
            # 如果有动作输入，计算对应的Q值
            if not isinstance(action, torch.Tensor):
                action = torch.tensor([action], dtype=torch.float32).to(device)
            action = action.view(-1, 1)  # 确保action的形状是 [batch_size, 1]
            
            # 计算Q值: Q(s,a) = V(s) + A(s,a)
            # 这里我们使用简单的线性优势函数: A(s,a) = a * advantage
            q_value = state_value + action * advantage
            return q_value.squeeze()  # 移除多余的维度
        else:
            # 如果没有动作输入，返回状态值和优势函数的参数
            return state_value.squeeze(), advantage.squeeze()

# 定义DQN_GNN类(定义Q网络以及一个固定的Q网络)
class DQN_GNN(object):
    def __init__(self, node_num, env_information):
        self.node_num = node_num  # 点总数
        self.env_information = env_information  # 环境信息
        # 创建评估网络和目标网络
        self.eval_net,self.target_net = Net(act_dim=1, node_num=self.node_num).to(device),Net(act_dim=1, node_num=self.node_num).to(device) 
        # self.eval_net,self.target_net = Net(2).to('cuda'),Net(2).to('cuda')
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0      # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY,6)) # 存储空间初始化，每一组的数据为(s_t,a_t,r_t,s_{t+1})
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # 使用MSE损失函数
        self.loss_func = nn.MSELoss()
        self.loss_func = self.loss_func.to(device)  # 将损失函数移动到GPU
        self.tau = 0.005  # 软更新参数

    def choose_action(self, episode_num, obs, x_graph, n_samples=21):
        """
        选择动作，使用Dueling DQN结构，支持连续动作空间
        episode_num: 当前周期数
        obs: 观察值
        x_graph: 图结构数据
        n_samples: 采样数量，用于在连续空间中寻找最佳动作
        return: 选择的动作，范围在[-1, 1]之间
        """
        epsilon = max(0.1, 0.90 - episode_num * 0.0001)  # 探索率衰减
        
        if np.random.uniform() > epsilon:  
            # 在[-1, 1]区间内均匀采样n_samples个点
            actions = np.linspace(-1, 1, n_samples)
            q_values = []
            
            # 计算每个动作的Q值
            with torch.no_grad():
                # 获取状态值和优势函数参数
                state_value, advantage = self.eval_net(x=obs[0], state=obs[1], x_graph=x_graph)
                
                # 转换为numpy数组以便计算

                state_value = state_value.cpu().numpy()
                advantage = advantage.cpu().numpy()
                
                # 计算每个动作的Q值
                for a in actions:
                    q = state_value + a * np.abs(advantage)
                    q_values.append(q)
            #print('values',q_values)
            # 选择Q值最大的动作
            best_action = actions[np.argmax(q_values)]
            #print('选择数组中动作')
            return float(best_action)  # 确保返回Python标量
        else:  # 探索
            # 在[-1, 1]区间内随机采样一个动作
            #print('随机选择动作')
            return float(np.random.uniform(-1, 1))
  
    def store_transition(self, o, s, a, r, o_, s_):  # 存储经验
        # This function acts as experience replay buffer
        s = [s]  # 将状态s转换为列表
        s_ = [s_]  # 将状态s_转换为列表
        transition = np.hstack((o, s, [a, r], o_, s_))  # 水平堆叠这些向量
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY  # 计算索引
        self.memory[index, :] = transition  # 将transition存储到memory中
        self.memory_counter += 1  # 记忆量计数器自加1

    def soft_update(self, target, source):
        """
        软更新目标网络参数
        target: 目标网络
        source: 源网络
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, rpm):
        """
        更新网络参数
        rpm: 经验回放池
        return: 损失值
        """
        # 软更新目标网络
        self.soft_update(self.target_net, self.eval_net)
        
        # 定期完全同步目标网络（可选）
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("完全更新目标网络参数")
        
        self.learn_step_counter += 1
        
        # 从经验回放池中采样
        b_o, b_s, b_a, b_r, b_o_, b_s_, done = rpm.sample(32)# 从经验回放缓存中随机采样64个样本
        x_graph = b_s
        edge_index_graph = self.eval_net.create_edge_index()
        loss_all = 0  # 初始化损失值
        for i in range(32):
            # 获得32个trasition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
            state = [b_s[i][1], b_s[i][0], b_s[i][5], b_s[i][4]]
            #q_eval = self.eval_net(b_o[i], b_s[i])[int(b_a[i])]  # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
            q_eval = self.eval_net(b_o[i], state, x_graph[i])[int(b_a[i])]  # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
            # q_next 不进行反向传播，故用detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
            #q_next = self.target_net(b_o_[i], b_s_[i])
            # q_next = self.target_net(b_o_[i], state, x_graph[i])
            # # 先算出目标值q_target，max(1)[0]相当于计算出每一行中的最大值（注意不是上面的索引了,而是一个一维张量了），view()函数让其变成(32,1)
            # q_target = b_r[i] + GAMMA * q_next.max(0)[0]
            state_value, advantage = self.target_net(b_o_[i], state, x_graph[i])
            q_next = state_value + torch.abs(advantage)  # 使用状态值和优势值计算Q值
            q_target = b_r[i] + GAMMA * q_next
            # 计算损失值
            loss = self.loss_func(q_eval, q_target)  # 计算损失值
            loss_all = loss_all + loss  # 累加损失值
        loss_all = loss_all / 32  # 计算平均损失值
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss_all.backward()  # 误差方向传播
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()  # 逐步的梯度优化
        return loss_all