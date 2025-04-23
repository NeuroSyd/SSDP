import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm  # 进度条
from torchvision import datasets, transforms

# 设置随机种子函数
def set_random_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用设置随机种子
SEED = 310
set_random_seed(SEED)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义全局变量，用于自定义激活函数的梯度计算
gamma = 1.0
lens = 0.5
scale = 6.0
hight = 0.15

# 自定义激活函数（可选）
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        if surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                   - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                   - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type == 'G':
            temp = gaussian(input, mu=0., sigma=lens)
        elif surrograte_type == 'linear':
            temp = F.relu(1 - input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5 * input.abs())
        elif surrograte_type == 'rect':
            temp = (input.abs() < 0.5).float()
        else:
            temp = torch.zeros_like(input)
        return grad_input * temp.float() * gamma

# 选择替代梯度类型
surrograte_type = 'MG'

def gaussian(x, mu=0., sigma=.5):
    """高斯函数，用于自定义激活函数的梯度计算"""
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(
        2 * torch.tensor(math.pi, device=x.device)) / sigma

act_fun_adp = ActFun_adp.apply

# SSDPModule 定义
class SSDPModule(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(SSDPModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 参数化的SSDP学习率，确保为标量
        self.A_plus = nn.Parameter(torch.tensor(0.02, device=device, dtype=torch.float32))
        self.A_minus = nn.Parameter(torch.tensor(0.02, device=device, dtype=torch.float32))
        self.tau_plus = nn.Parameter(torch.tensor(20.0, device=device, dtype=torch.float32))
        self.tau_minus = nn.Parameter(torch.tensor(20.0, device=device, dtype=torch.float32))

    def forward(self, pre_spike, post_spike, delta_t):
        """
        pre_spike: [batch_size, input_dim]
        post_spike: [batch_size, output_dim]
        delta_t: [batch_size, 1]
        Returns:
            delta_w: [output_dim, input_dim]
        """
        # 扩展维度以便进行广播
        post_spike_expanded = post_spike.unsqueeze(-1)  # [batch_size, output_dim, 1]
        pre_spike_expanded = pre_spike.unsqueeze(-2)    # [batch_size, 1, input_dim]
        delta_t_expanded = (delta_t / self.tau_plus).unsqueeze(-1)  # [batch_size, 1, 1]

        # 计算权重增加（potentiation）
        delta_w_pot = self.A_plus * post_spike_expanded * pre_spike_expanded * torch.exp(
            -delta_t_expanded)  # [batch_size, output_dim, input_dim]

        # 计算权重减少（depression）
        delta_w_dep = self.A_minus * post_spike_expanded * pre_spike_expanded * torch.exp(
            -delta_t_expanded / self.tau_minus)  # [batch_size, output_dim, input_dim]

        # 计算权重更新，平均所有批次
        delta_w = (delta_w_pot - delta_w_dep).mean(dim=0)  # [output_dim, input_dim]

        # 限制 delta_w 的范围为 -1 到 1
        delta_w = torch.clamp(delta_w, min=-1.0, max=1.0)

        return delta_w

# Spike RNN层（集成了SSDP）
class SpikeRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, tau_minitializer='uniform', low_m=2, high_m=6,
                 tau_ninitializer='uniform', low_n=2, high_n=6, vth=0.5, dt=1, branch=1, device='cuda', bias=True):
        super(SpikeRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.branch = branch
        self.device = device
        self.vth = vth
        self.dt = dt

        # 输入和隐藏层的全连接层
        self.dense_input = nn.Linear(input_dim, hidden_dim * branch)
        self.layernorm_input = nn.LayerNorm(hidden_dim * branch)  # 添加 LayerNorm
        self.dense_hidden = nn.Linear(hidden_dim, hidden_dim * branch, bias=bias)
        self.layernorm_hidden = nn.LayerNorm(hidden_dim * branch)  # 添加 LayerNorm

        nn.init.xavier_normal_(self.dense_input.weight)
        nn.init.xavier_normal_(self.dense_hidden.weight)
        if bias:
            nn.init.constant_(self.dense_input.bias, 0.0)
            nn.init.constant_(self.dense_hidden.bias, 0.0)

        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_n = nn.Parameter(torch.Tensor(hidden_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

        # 实例化输入和隐藏层各自的 SSDPModule，传递 device
        self.ssdp_input = SSDPModule(input_dim=input_dim, output_dim=hidden_dim * branch, device=device)
        self.ssdp_hidden = SSDPModule(input_dim=hidden_dim, output_dim=hidden_dim, device=device)

        # 初始化阈值为张量，确保它与隐藏维度匹配
        self.v_th = torch.ones(1, hidden_dim, device=self.device) * self.vth

        # 添加用于保存前一个脉冲的属性
        self.prev_spike = None

    def forward(self, input_spike_sequence):
        """
        input_spike_sequence: [batch_size, seq_length, input_dim]
        Returns:
            mem_sequence: [batch_size, seq_length, hidden_dim]
            spike_sequence: [batch_size, seq_length, hidden_dim]
        """
        batch_size, seq_length, input_dim = input_spike_sequence.size()
        mem_sequence = []
        spike_sequence = []

        # 初始化内部状态
        mem = torch.rand(batch_size, self.hidden_dim, device=self.device)
        d_input = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        self.prev_spike = torch.zeros(batch_size, self.hidden_dim, device=self.device)  # 跟踪前一个脉冲

        for t in range(seq_length):
            input_spike = input_spike_sequence[:, t, :]
            alpha = torch.exp(-self.dt / torch.clamp(self.tau_m, min=1e-2))  # [hidden_dim]
            alpha = alpha.unsqueeze(0)  # [1, hidden_dim]

            # 通过全连接层获取输入和隐藏层的电流，并应用 LayerNorm
            input_current = self.dense_input(input_spike)
            input_current = self.layernorm_input(input_current)
            input_current = input_current.reshape(batch_size, self.hidden_dim, self.branch)

            hidden_current = self.dense_hidden(self.prev_spike)
            hidden_current = self.layernorm_hidden(hidden_current)
            hidden_current = hidden_current.reshape(batch_size, self.hidden_dim, self.branch)

            # 计算 (input_current + hidden_current) 的总和，沿着分支维度
            combined_current = (input_current + hidden_current).sum(dim=2)  # [batch_size, hidden_dim]

            # 更新 dendritic input
            d_input = torch.sigmoid(self.tau_n).unsqueeze(0) * d_input + \
                      (1 - torch.sigmoid(self.tau_n)).unsqueeze(0) * combined_current  # [batch_size, hidden_dim]

            # 更新膜电位
            mem = alpha * mem + (1 - alpha) * d_input  # [batch_size, hidden_dim]

            # 计算脉冲，使用自定义激活函数
            inputs_ = mem - self.v_th  # [batch_size, hidden_dim]
            spike = act_fun_adp(inputs_)  # 使用自定义激活函数

            mem_sequence.append(mem)
            spike_sequence.append(spike)

            # 更新前一个脉冲
            self.prev_spike = spike

        mem_sequence = torch.stack(mem_sequence, dim=1)  # [batch_size, seq_length, hidden_dim]
        spike_sequence = torch.stack(spike_sequence, dim=1)  # [batch_size, seq_length, hidden_dim]

        # 返回 mem_sequence 和 spike_sequence
        return mem_sequence, spike_sequence

# 读出层
class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', bias=True):
        super(ReadoutLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.layernorm = nn.LayerNorm(output_dim)  # 添加 LayerNorm

        nn.init.xavier_normal_(self.dense.weight)
        if bias:
            nn.init.constant_(self.dense.bias, 0.0)

    def forward(self, spike_sequence):
        """
        spike_sequence: [batch_size, seq_length, input_dim]
        """
        # 聚合脉冲序列，例如对时间维度取平均
        aggregated_spike = spike_sequence.mean(dim=1)  # [batch_size, input_dim]
        output = self.dense(aggregated_spike)  # [batch_size, output_dim]
        output = self.layernorm(output)  # 应用 LayerNorm
        return output

# 完整的SNN模型
class SNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, branch=1, vth=1.0, dt=1, device='cuda'):
        super(SNNModel, self).__init__()
        self.rnn = SpikeRNN(input_dim, hidden_dim, branch=branch, device=device)
        self.readout = ReadoutLayer(hidden_dim, output_dim, device=device)

    def forward(self, input):
        mem, spike = self.rnn(input)  # 返回 SSDP 所需的数据
        output = self.readout(spike)  # ReadoutLayer的输出
        return output, spike

    def reset(self, batch_size):
        pass

# 定义Dataset类
class CIFAR10_SNN_Dataset(Dataset):
    def __init__(self, train=True, transform=None, seq_length=50, encoding_rate=20.0, modulation='exponential'):
        """
        参数:
            train (bool): 是否为训练集。
            transform (callable, optional): 对图像进行预处理的转换。
            seq_length (int): 脉冲序列的时间步数。
            encoding_rate (float): 基础脉冲生成率。
            modulation (str): 调制函数类型，支持 'linear' 和 'exponential'。
        """
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        self.transform = transform
        self.seq_length = seq_length
        self.encoding_rate = encoding_rate  # 基础脉冲生成概率
        self.modulation = modulation  # 调制函数类型

        # 预计算调制因子
        if self.modulation == 'linear':
            # 线性衰减
            self.modulation_factors = torch.linspace(1.0, 0.0, steps=self.seq_length)
        elif self.modulation == 'exponential':
            # 指数衰减
            decay_rate = 5.0  # 调节衰减速度
            self.modulation_factors = torch.exp(-torch.linspace(0, decay_rate, steps=self.seq_length))
        else:
            raise ValueError("Unsupported modulation type. Choose 'linear' or 'exponential'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)  # [3, 32, 32]
        # 将图像展开为一维向量
        img = img.view(-1)  # [3072]

        # 使用时间依赖的泊松编码将静态图像转换为脉冲序列
        img_sequence = []
        for t in range(self.seq_length):
            modulation_factor = self.modulation_factors[t]
            spike_prob = img * self.encoding_rate * modulation_factor  # 动态调整脉冲生成概率
            spike_prob = torch.clamp(spike_prob, 0, 1)  # 确保概率在 [0,1] 之间
            spike = torch.bernoulli(spike_prob)
            img_sequence.append(spike)
        img_sequence = torch.stack(img_sequence, dim=0)  # [seq_length, 3072]

        return img_sequence, label

def cosine_annealing(epoch, T_max, eta_max, eta_min=0):
    """计算余弦退火学习率"""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max))

def main():
    # 调用设置随机种子
    SEED = 310
    set_random_seed(SEED)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数
    input_dim = 3 * 32 * 32  # CIFAR-10的输入维度
    hidden_dim = 1024  #
    output_dim = 10  # CIFAR-10的输出维度
    batch_size = 256  #
    num_epochs = 100
    learning_rate = 0.01
    seq_length = 10  # 定义脉冲序列的时间步数

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量 [3, 32, 32]，范围 [0,1]
        # 可以添加更多预处理步骤，如归一化
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载CIFAR-10数据集
    train_dataset = CIFAR10_SNN_Dataset(train=True, transform=transform, seq_length=seq_length)
    valid_dataset = CIFAR10_SNN_Dataset(train=False, transform=transform, seq_length=seq_length)
    test_dataset = CIFAR10_SNN_Dataset(train=False, transform=transform, seq_length=seq_length)

    # 使用DataLoader加载训练、验证和测试集，暂时将num_workers=2
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=2)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # 初始化模型
    model = SNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        branch=1,  # 确保分支数为1
        vth=0.5,  # 降低阈值以增加激活率
        dt=1,
        device=device
    ).to(device)

    # 优化器
    optimizer = optim.Adam([
        {'params': model.readout.parameters()},
        {'params': model.rnn.dense_input.parameters()},
        {'params': model.rnn.dense_hidden.parameters()},
        {'params': model.rnn.tau_m},
        {'params': model.rnn.tau_n},
        # 不包含 SSDPModule 的参数
    ], lr=learning_rate, weight_decay=1e-5)

    # 添加余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    # 损失函数和其他参数
    criterion = nn.CrossEntropyLoss()
    lambda_fr = 0.5  # 减少激活率正则化权重
    target_firing_rate = 0.1

    # 训练循环
    for epoch in range(num_epochs):
        # 更新 SSDP 参数
        current_A_plus = cosine_annealing(epoch, num_epochs, eta_max=0.02, eta_min=0.0001)
        current_A_minus = cosine_annealing(epoch, num_epochs, eta_max=0.02, eta_min=0.0001)
        model.rnn.ssdp_input.A_plus.data.fill_(current_A_plus)
        model.rnn.ssdp_input.A_minus.data.fill_(current_A_minus)
        model.rnn.ssdp_hidden.A_plus.data.fill_(current_A_plus)
        model.rnn.ssdp_hidden.A_minus.data.fill_(current_A_minus)

        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            labels = labels.view(-1).long()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 获取当前批次的实际批量大小
            current_batch_size = inputs.size(0)

            # 确保输入的形状为 [batch_size, seq_length, input_dim]
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)  # [batch_size, 1, input_dim]

            # 前向传播
            output, spike = model(inputs)

            # 计算损失
            loss_ce = criterion(output, labels)  # 交叉熵损失
            firing_rate = spike.mean()
            loss_fr = (firing_rate - target_firing_rate) ** 2  # 激活率正则化
            total_loss = loss_ce + lambda_fr * loss_fr

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 手动更新权重
            with torch.no_grad():
                # 获取最后一个时间步的输入和脉冲
                last_input_spike = inputs[:, -1, :]  # [current_batch_size, input_dim]
                last_output_spike = spike[:, -1, :]  # [current_batch_size, hidden_dim]

                # 创建 delta_t，使用当前批量大小
                delta_t = torch.ones(current_batch_size, 1, device=device)

                # 更新 dense_input 的权重
                delta_w_input = model.rnn.ssdp_input(
                    pre_spike=last_input_spike,
                    post_spike=last_output_spike,
                    delta_t=delta_t
                )
                if delta_w_input.shape == model.rnn.dense_input.weight.shape:
                    model.rnn.dense_input.weight += delta_w_input
                else:
                    print("delta_w_input shape does not match dense_input.weight shape. Skipping update for dense_input.")

                # 更新 dense_hidden 的权重
                delta_w_hidden = model.rnn.ssdp_hidden(
                    pre_spike=model.rnn.prev_spike,  # 上一个时间步的脉冲
                    post_spike=last_output_spike,
                    delta_t=delta_t
                )
                if delta_w_hidden.shape == model.rnn.dense_hidden.weight.shape:
                    model.rnn.dense_hidden.weight += delta_w_hidden
                else:
                    print("delta_w_hidden shape does not match dense_hidden.weight shape. Skipping update for dense_hidden.")

            # 优化器步骤
            optimizer.step()

            running_loss += loss_ce.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, Firing Rate: {firing_rate.item():.4f}, '
                      f'Firing Rate Loss: {loss_fr.item():.4f}')
                running_loss = 0.0

        # 更新学习率
        scheduler.step()
        # 打印当前学习率
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')

    # 测试模型
    def test_classifier(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                labels = labels.view(-1).long()
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # 确保输入的形状为 [batch_size, seq_length, input_dim]
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)  # [batch_size, 1, input_dim]

                # 前向传播
                output, spike = model(inputs)
                _, predicted = torch.max(output, 1)  # 获取预测结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy of the model on the test set: {accuracy:.2f}%')

    test_classifier(model, test_loader)  # 在测试集上测试

if __name__ == '__main__':
    # 启用异常检测以获得更详细的错误信息
    torch.autograd.set_detect_anomaly(True)
    main()
