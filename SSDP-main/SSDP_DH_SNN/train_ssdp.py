import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from sklearn.metrics import confusion_matrix
import scipy.io
import random
import os
from shd_dataset import my_Dataset
from tqdm import tqdm  # 进度条

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)

# 数据集和 DataLoader（使用 os.path.join 拼接路径）
dt = 1
batch_size = 512
train_dir = '/mnt/data_pci_2_2T/viktor/CSNASNet_SHD/data/SHD/train_1ms'
train_files = [os.path.join(train_dir, i) for i in os.listdir(train_dir)]
test_dir = '/mnt/data_pci_2_2T/viktor/CSNASNet_SHD/data/SHD/test_1ms'
test_files = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
train_dataset = my_Dataset(train_files)
test_dataset = my_Dataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 导入 SNN 模块（确保这些模块在你的环境中可用）
from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *

thr_func = ActFun_adp.apply
is_bias = True


# ===================== 为 readout_integrator_test 添加 forward_with_spike =====================
# 如果原类没有此接口，则通过继承包装
class ReadoutIntegratorWithSpike(readout_integrator_test):
    def forward_with_spike(self, x):
        """
        调用原始 forward 得到膜电位 mem，
        并简单地以 0 为阈值产生脉冲 spike
        """
        mem = self.forward(x)
        spike = (mem > 0).float()
        return mem, spike


# ===================== SSDPModule 定义（增加初始值保存，用于调度） =====================
class SSDPModule(nn.Module):
    def __init__(self, input_dim, output_dim, device,
                 A_plus=0.00015, A_minus=0.0001, A_baseline=0.00005, sigma=1.0):
        super(SSDPModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A_plus = nn.Parameter(torch.tensor(A_plus, device=device, dtype=torch.float32))
        self.A_minus = nn.Parameter(torch.tensor(A_minus, device=device, dtype=torch.float32))
        self.A_baseline = nn.Parameter(torch.tensor(A_baseline, device=device, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, device=device, dtype=torch.float32))
        # 保存初始值，便于调度时进行缩放
        self.init_A_plus = self.A_plus.data.clone()
        self.init_A_minus = self.A_minus.data.clone()
        self.init_A_baseline = self.A_baseline.data.clone()
        self.init_sigma = self.sigma.data.clone()

    def forward(self, pre_spike, post_spike, delta_t):
        """
        pre_spike: [B, C_in] —— dense_2 输入端脉冲状态
        post_spike: [B, C_out] —— dense_2 输出端脉冲状态
        delta_t: [B, C_out, C_in] —— 每个样本中 t_post 与 t_pre 的绝对时间差
        """
        post_spike_expanded = post_spike.unsqueeze(-1)  # [B, C_out, 1]
        pre_spike_expanded = pre_spike.unsqueeze(1)  # [B, 1, C_in]
        synchronized = post_spike_expanded * pre_spike_expanded  # [B, C_out, C_in]
        gauss = torch.exp(- (delta_t ** 2) / (2 * (self.sigma ** 2)))
        delta_w_pot = self.A_plus * synchronized * gauss
        delta_w_dep = self.A_baseline * (1 - synchronized) * gauss
        delta_w = (delta_w_pot - delta_w_dep).mean(dim=0)
        delta_w = torch.clamp(delta_w, -1.0, 1.0)
        return delta_w


# ===================== SSDP 参数余弦退火调度器 =====================
def update_ssdp_params_cos(ssdp_module, epoch, start_epoch, T_max, eta_min):
    """
    当 epoch >= start_epoch 后，使用余弦退火对 SSDP 参数进行更新。
    参数计算公式：
       new_val = init_val × [eta_min + 0.5*(1 - eta_min)*(1 + cos(pi * progress))]
    其中 progress = (epoch - start_epoch) / T_max
    T_max: 总进度（例如 epochs - start_epoch）
    eta_min: 最小比例（如 0 表示退火到 0）
    """
    if epoch < start_epoch:
        return
    progress = (epoch - start_epoch) / T_max
    cosine_factor = eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))
    new_A_plus = ssdp_module.init_A_plus * cosine_factor
    new_A_minus = ssdp_module.init_A_minus * cosine_factor
    new_A_baseline = ssdp_module.init_A_baseline * cosine_factor
    new_sigma = ssdp_module.init_sigma * cosine_factor
    with torch.no_grad():
        ssdp_module.A_plus.copy_(new_A_plus)
        ssdp_module.A_minus.copy_(new_A_minus)
        ssdp_module.A_baseline.copy_(new_A_baseline)
        ssdp_module.sigma.copy_(new_sigma)
    print(f"Epoch {epoch}: Updated SSDP params (cos annealing): "
          f"A_plus {ssdp_module.A_plus.item():.6f}, A_minus {ssdp_module.A_minus.item():.6f}, "
          f"A_baseline {ssdp_module.A_baseline.item():.6f}, sigma {ssdp_module.sigma.item():.6f}")



# ===================== 集成 SSDP 的 DHSNN 模型定义 =====================
class rnn_test(nn.Module):
    def __init__(self, device):
        super(rnn_test, self).__init__()
        n = 128  # 这里 n 就代表隐藏层神经元的个数
        self.n = n
        # 创建 DH-SRNN 层（输入维度700，输出维度n，即隐藏层神经元数）
        self.rnn_1 = spike_rnn_test_denri_wotanh_R(700, n, tau_ninitializer='uniform',
                                                   low_n=2, high_n=6, vth=1, dt=1, branch=8, device=device)
        # 创建读出层，用包装后的 ReadoutIntegratorWithSpike 替换原来的读出层
        self.dense_2 = ReadoutIntegratorWithSpike(n, 20, dt=1, device=device)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
        if is_bias:
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

        # 集成 SSDP 模块（作用于 dense_2 层，其 input_dim 对应 n，output_dim 对应20）
        self.ssdp = SSDPModule(input_dim=n, output_dim=20, device=device)
        # 设定 SSDP 更新起始的 epoch
        self.start_ssdp_epoch = 10

        # 用于记录 dense_2 输入（pre）的脉冲首次时刻（形状 [B, n]）
        self.t_pre = None
        # 用于记录 dense_2 输出（post）的脉冲首次时刻（形状 [B, 20]）
        self.t_post = None

    def forward(self, input):
        """
        input: [B, seq_length, input_dim]
        返回：累积的分类输出、rnn_1 的脉冲状态（用于 dense_2 输入）以及 dense_2 的脉冲状态
        同时记录各层第一次发脉冲的时间
        """
        input = input.to(next(self.parameters()).device)
        b, seq_length, input_dim = input.shape

        self.rnn_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)

        output = 0
        pre_spike_list = []
        post_spike_list = []
        t_pre = None  # [B, n]
        t_post = None  # [B, 20]

        for t in tqdm(range(seq_length), desc="Time steps", leave=False):
            input_t = input[:, t, :].reshape(b, input_dim)
            mem_layer1, spike_layer1 = self.rnn_1(input_t)
            mem_layer2, spike_layer2 = self.dense_2.forward_with_spike(spike_layer1)

            if t > 10:
                output += F.softmax(mem_layer2, dim=1)

            pre_spike_list.append(spike_layer1)
            post_spike_list.append(spike_layer2)

            if t == 0:
                t_pre = torch.full((b, spike_layer1.size(1)), float(seq_length), device=spike_layer1.device)
                t_post = torch.full((b, spike_layer2.size(1)), float(seq_length), device=spike_layer2.device)

            new_spike_pre = (spike_layer1 > 0).float()
            new_spike_post = (spike_layer2 > 0).float()
            t_pre = torch.where((t_pre == float(seq_length)) & (new_spike_pre == 1),
                                torch.tensor(float(t), device=spike_layer1.device),
                                t_pre)
            t_post = torch.where((t_post == float(seq_length)) & (new_spike_post == 1),
                                 torch.tensor(float(t), device=spike_layer2.device),
                                 t_post)

        pre_spike = (torch.stack(pre_spike_list, dim=0).sum(dim=0) > 0).float()  # [B, n]
        post_spike = (torch.stack(post_spike_list, dim=0).sum(dim=0) > 0).float()  # [B, 20]

        self.t_pre = t_pre
        self.t_post = t_post

        return output, pre_spike, post_spike


# ===================== 模型实例化、损失函数及设备转换 =====================
model = rnn_test(device)
criterion = nn.CrossEntropyLoss()
print("device:", device)
model.to(device)


# ===================== 测试函数 =====================
def test():
    test_acc = 0.
    sum_sample = 0.
    model.eval()
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        model.rnn_1.apply_mask()
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        predictions, _, _ = model(images)
        _, predicted = torch.max(predictions.data, 1)
        test_acc += (predicted.cpu() == labels.cpu()).sum().item()
        sum_sample += predicted.cpu().numel()
    return test_acc / sum_sample


# ===================== 训练函数 =====================
def train(epochs, criterion, optimizer, scheduler=None):
    acc_list = []
    best_acc = 0
    path = 'model/'  # 模型保存路径
    name = 'rnn_denri_branch8_1ms_final_64neuron_MG_bs100_nl2h6_woclipnorm_initzeros_seed0'
    # 在每个 epoch 开始前设定 SSDP 参数调度器的参数（例如 step_size=20, gamma=0.5）
    ssdp_step_size = 20
    ssdp_gamma = 0.5
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        model.train()
        model.rnn_1.apply_mask()
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            optimizer.zero_grad()
            predictions, pre_spike, post_spike = model(images)
            _, predicted = torch.max(predictions.data, 1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            model.rnn_1.apply_mask()

            train_loss_sum += loss.item()
            train_acc += (predicted.cpu() == labels.cpu()).sum().item()
            sum_sample += predicted.numel()

            if epoch >= model.start_ssdp_epoch:
                t_post = model.t_post.unsqueeze(2)  # [B, 20, 1]
                t_pre = model.t_pre.unsqueeze(1)  # [B, 1, 64]
                delta_t = (t_post - t_pre).abs()  # [B, 20, 64]
                with torch.no_grad():
                    delta_w = model.ssdp(pre_spike, post_spike, delta_t)
                    if delta_w.shape == model.dense_2.dense.weight.shape:
                        model.dense_2.dense.weight.add_(delta_w)
                    else:
                        print("SSDP weight update shape mismatch:", delta_w.shape, model.dense_2.dense.weight.shape)

        if scheduler:
            scheduler.step()
        # 更新 SSDP 参数
        update_ssdp_params_cos(model.ssdp, epoch, model.start_ssdp_epoch, ssdp_step_size, ssdp_gamma)

        train_acc = train_acc / sum_sample
        valid_acc = test()
        print('Epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}'.format(
            epoch, train_loss_sum / len(train_loader), train_acc, valid_acc), flush=True)
        acc_list.append(train_acc)
    return acc_list


# ===================== 优化器、学习率调度器及训练调用 =====================
learning_rate = 1e-2
base_params = [
    model.dense_2.dense.weight,
    model.dense_2.dense.bias,
    model.rnn_1.dense.weight,
    model.rnn_1.dense.bias,
]
optimizer = torch.optim.Adam([
    {'params': base_params, 'lr': learning_rate},
    {'params': model.dense_2.tau_m, 'lr': learning_rate * 2},
    {'params': model.rnn_1.tau_m, 'lr': learning_rate * 2},
    {'params': model.rnn_1.tau_n, 'lr': learning_rate * 2},
], lr=learning_rate)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
epochs = 100

acc_list = train(epochs, criterion, optimizer, scheduler)
test_acc = test()
print("Test Accuracy: ", test_acc)
