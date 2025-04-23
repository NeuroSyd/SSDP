import numpy as np
import matplotlib.pyplot as plt

# 加载 spike 数据
spikes_withSSDP = np.load("spikes_record_SSDP.npy")      # shape: [N, T, C]
spikes_withoutSSDP = np.load("spikes_record.npy")        # shape: [N, T, C]

# jitter 分析函数
def compute_jitter_random(spike_tensor, neuron_num=100, seed=42):
    np.random.seed(seed)  # 确保可复现
    N, T, C = spike_tensor.shape
    neuron_indices = np.arange(C)
    np.random.shuffle(neuron_indices)  # 打乱神经元索引
    jitter_per_neuron = []
    count = 0

    for neuron in neuron_indices:
        spike_times = []
        for n in range(N):
            firing = spike_tensor[n, :, neuron]
            time_indices = np.where(firing > 0)[0]
            if len(time_indices) > 0:
                spike_times.append(time_indices.mean())
        if len(spike_times) > 1:
            jitter = np.std(spike_times)
            jitter_per_neuron.append(jitter)
            count += 1
            if count >= neuron_num:
                break
    return np.array(jitter_per_neuron)

# 使用示例：
jitter_ssdp = compute_jitter_random(spikes_withSSDP, neuron_num=300)
jitter_nossdp = compute_jitter_random(spikes_withoutSSDP, neuron_num=300)


print(f"SSDP analyzed neurons: {len(jitter_ssdp)}")
print(f"Backprop-only analyzed neurons: {len(jitter_nossdp)}")

# 绘图
plt.figure(figsize=(14, 8))
plt.hist(jitter_nossdp, bins=30, alpha=0.6, label='Without SSDP', color='steelblue')
plt.hist(jitter_ssdp, bins=30, alpha=0.6, label='SSDP', color='orange')
plt.xlabel("Spike Time Jitter",fontsize=36,fontweight='bold')
plt.ylabel("Number of Neurons",fontsize=36,fontweight='bold')
plt.xticks(fontsize=30,fontweight='bold')
plt.yticks(fontsize=30,fontweight='bold')
# plt.title("Spike Time Jitter Distribution")
plt.legend(fontsize=30, loc='best', prop={'size': 30, 'weight': 'bold'})
# plt.grid(True)
plt.tight_layout()
plt.show()
