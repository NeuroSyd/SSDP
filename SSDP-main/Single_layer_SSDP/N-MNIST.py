import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import math
import torch.nn.functional as F
import random
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# 超参数
input_dim = 34*34  # 1156
hidden_dim = 1500
output_dim = 10
batch_size = 512
num_epochs = 100
learning_rate = 0.001
seq_length = 50  # 根据N-MNIST的实际序列长度调整

# SSDP参数（保持不变或根据需要调整）
gamma = 0.5
lens = 0.5
scale = 6.0
hight = 0.15

# 高斯函数，用于自定义激活函数的梯度计算
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(
        2 * torch.tensor(math.pi, device=x.device)) / sigma

# 自定义激活函数
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
                 tau_ninitializer='uniform', low_n=2, high_n=6, vth=0.5, dt=1, branch=1, device='cuda', bias=True, dropout_p=0.5):
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

        # 添加 Dropout
        self.dropout = nn.Dropout(p=dropout_p)

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
            combined_current = self.dropout(combined_current)  # 应用 Dropout

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
    def __init__(self, input_dim, hidden_dim, output_dim, branch=1, vth=1.0, dt=1, device='cuda', num_layers=1, dropout_p=0.5):
        super(SNNModel, self).__init__()
        self.rnn = SpikeRNN(input_dim, hidden_dim, branch=branch, device=device, dropout_p=dropout_p)
        self.readout = ReadoutLayer(hidden_dim, output_dim, device=device)

    def forward(self, input):
        mem, spike = self.rnn(input)
        output = self.readout(spike)
        return output, spike

    def reset(self, batch_size):
        pass

# 定义N-MNIST的Dataset类（已包含聚合方式）
class NMNISTDataset(Dataset):
    def __init__(self, data_dir, seq_length=10, transform=None, aggregation='binary'):
        """
        data_dir: 存放N-MNIST数据集的根目录。
        seq_length: 脉冲序列的长度（时间步数）。
        transform: 数据转换操作。
        aggregation: 事件聚合方式，'binary' 或 'count'。
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.transform = transform
        self.aggregation = aggregation  # 新增参数
        self.samples = self._load_file_paths()

    def _load_file_paths(self):
        """
        加载数据集中的所有样本文件路径，并从文件夹名称提取标签。
        """
        samples = []
        for label_folder in os.listdir(self.data_dir):
            label_folder_path = os.path.join(self.data_dir, label_folder)
            if os.path.isdir(label_folder_path):
                try:
                    label = int(label_folder)  # 子文件夹名为标签（0-9）
                except ValueError:
                    print(f"Skipping non-integer folder: {label_folder}")
                    continue
                for file in sorted(os.listdir(label_folder_path)):
                    if file.endswith(".bin"):  # 假设事件文件扩展名是.bin
                        file_path = os.path.join(label_folder_path, file)
                        samples.append((file_path, label))
        return samples

    def _parse_events(self, file_path, aggregation=None):
        """
        从二进制文件中读取事件信息并转换为脉冲张量。
        """
        if aggregation is None:
            aggregation = self.aggregation

        try:
            with open(file_path, 'rb') as f:
                events = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 5)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return torch.zeros((self.seq_length, 34, 34), dtype=torch.float32)

        # 提取事件数据
        x = events[:, 0]
        y = events[:, 1]
        polarity = (events[:, 2] >> 7) & 1  # 提取极性位
        timestamp = ((events[:, 2] & 0x7F) << 16) | (events[:, 3] << 8) | events[:, 4]  # 提取时间戳

        # 初始化脉冲序列：维度为 [seq_length, 34, 34]
        spike_tensor = torch.zeros((self.seq_length, 34, 34), dtype=torch.float32)

        # 时间窗口设置
        if len(timestamp) == 0:
            return spike_tensor  # 如果没有事件，返回全零张量

        time_bin = (timestamp.max() - timestamp.min()) / self.seq_length if timestamp.max() != timestamp.min() else 1
        for t in range(self.seq_length):
            # 获取属于当前时间窗的事件
            mask = (timestamp >= t * time_bin) & (timestamp < (t + 1) * time_bin)
            events_in_window = np.where(mask)[0]
            if len(events_in_window) == 0:
                continue  # 当前时间窗内无事件

            for i in events_in_window:
                # 确保 x 和 y 不超出范围
                if x[i] < 34 and y[i] < 34:
                    if aggregation == 'binary':
                        spike_tensor[t, y[i], x[i]] = 1.0  # 二值化处理
                    elif aggregation == 'count':
                        spike_tensor[t, y[i], x[i]] += 1.0  # 事件计数
                    elif aggregation == 'binary_polarity':
                        spike_tensor[t, y[i], x[i]] = polarity[i]  # 根据极性设置0或1
                    elif aggregation == 'binary_polarity_neg':
                        # 将极性映射为-1和1
                        spike_tensor[t, y[i], x[i]] = polarity[i] * 2 - 1
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregation}")

        # 对于 'count' 聚合方式，可以进行归一化处理
        if aggregation == 'count':
            spike_tensor = torch.clamp(spike_tensor, max=1.0)  # 例如，将计数上限设为1

        return spike_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        spike_tensor = self._parse_events(file_path)

        if self.transform:
            spike_tensor = self.transform(spike_tensor)

        # 将脉冲张量展平为 [seq_length, input_dim]
        spike_tensor = spike_tensor.view(self.seq_length, -1)  # [seq_length, 34*34]

        return spike_tensor, label

# 余弦退火学习率调度器
def cosine_annealing(epoch, T_max, eta_max, eta_min=0):
    """计算余弦退火学习率"""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max))

# 测试分类器
def test_classifier(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            labels = labels.view(-1).long()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # 确保输入的形状为 [batch_size, seq_length, input_dim]
            if inputs.dim() == 3:
                pass  # N-MNIST 数据已经是 [batch_size, seq_length, input_dim]
            elif inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)  # [batch_size, 1, input_dim]
            else:
                raise ValueError(f"Unexpected input dimensions: {inputs.dim()}")

            # 前向传播
            output, spike = model(inputs)
            _, predicted = torch.max(output, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy

# 绘制混淆矩阵
def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Confusion Matrix"):
            labels = labels.view(-1).long()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output, spike = model(inputs)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 训练与评估函数
def train_and_evaluate(aggregation_method):
    # 数据集目录
    data_dir = './data/NMNIST'  # 替换为您的N-MNIST数据集路径

    # 数据预处理（如果需要）
    transform = transforms.Compose([
        # 添加需要的转换，例如归一化等
    ])

    # 实例化数据集和数据加载器
    train_dataset = NMNISTDataset(
        data_dir=os.path.join(data_dir, 'Train'),
        seq_length=seq_length,
        transform=transform,
        aggregation=aggregation_method
    )
    val_dataset = NMNISTDataset(
        data_dir=os.path.join(data_dir, 'Train'),
        seq_length=seq_length,
        transform=transform,
        aggregation=aggregation_method
    )
    test_dataset = NMNISTDataset(
        data_dir=os.path.join(data_dir, 'Test'),
        seq_length=seq_length,
        transform=transform,
        aggregation=aggregation_method
    )

    # 划分训练集和验证集（例如80%训练，20%验证）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # 初始化模型
    model = SNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        branch=1,
        vth=0.5,
        dt=1,
        device=device,
        dropout_p=0.5
    ).to(device)

    # 优化器
    optimizer = optim.Adam([
        {'params': model.readout.parameters()},
        {'params': model.rnn.dense_input.parameters()},
        {'params': model.rnn.dense_hidden.parameters()},
        {'params': model.rnn.tau_m},
        {'params': model.rnn.tau_n},
    ], lr=learning_rate, weight_decay=1e-4)  # 增加 weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 损失函数和其他参数
    criterion = nn.CrossEntropyLoss()
    lambda_fr = 0.1  # 减少激活率正则化权重
    target_firing_rate = 0.2

    # 提前停止参数
    patience = 10  # 提前停止的耐心参数
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 更新 SSDP 参数（如果需要）
        current_A_plus = cosine_annealing(epoch, num_epochs, eta_max=0.02, eta_min=0.0001)
        current_A_minus = cosine_annealing(epoch, num_epochs, eta_max=0.02, eta_min=0.0001)
        model.rnn.ssdp_input.A_plus.data.fill_(current_A_plus)
        model.rnn.ssdp_input.A_minus.data.fill_(current_A_minus)
        model.rnn.ssdp_hidden.A_plus.data.fill_(current_A_plus)
        model.rnn.ssdp_hidden.A_minus.data.fill_(current_A_minus)

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            labels = labels.view(-1).long()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

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

            # 优化器步骤
            optimizer.step()

            running_loss += loss_ce.item()

            # 记录训练准确率
            _, predicted = torch.max(output, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / 50
                train_acc = 100 * correct_train / total_train
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, Firing Rate: {firing_rate.item():.4f}, '
                      f'Firing Rate Loss: {loss_fr.item():.4f}, Train Acc: {train_acc:.2f}%')
                running_loss = 0.0
                correct_train = 0
                total_train = 0

        # 验证过程
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.view(-1).long()
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # 前向传播
                output, spike = model(inputs)
                loss_ce = criterion(output, labels)
                firing_rate = spike.mean()
                loss_fr = (firing_rate - target_firing_rate) ** 2
                total_loss = loss_ce + lambda_fr * loss_fr
                val_loss += total_loss.item()

                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # 检查验证损失是否有所改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'best_model_{aggregation_method}.pth')
            print(f'Validation loss decreased, saving model.')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 使用验证损失更新调度器
        scheduler.step(val_loss)

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_{aggregation_method}.pth'))

    # 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, device)

    # 测试模型
    test_accuracy = test_classifier(model, test_loader, device)  # 在测试集上测试
    print(f'Test Accuracy with {aggregation_method} aggregation: {test_accuracy:.2f}%')

    return test_accuracy

# 主函数
def main():
    # 数据预处理（如果需要）
    transform = transforms.Compose([
        # 添加需要的转换，例如归一化等
    ])

    aggregation_methods = ['binary', 'count']  # 可以根据需要添加更多聚合方式
    results = {}

    for agg in aggregation_methods:
        print(f"\n--- Training with {agg} aggregation ---\n")
        accuracy = train_and_evaluate(aggregation_method=agg)
        results[agg] = accuracy

    print("\n--- Summary of Results ---")
    for agg, acc in results.items():
        print(f"Aggregation Method: {agg}, Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    # 启用异常检测以获得更详细的错误信息
    torch.autograd.set_detect_anomaly(True)
    main()
