import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import matplotlib.gridspec as gridspec

# ---------- 数据加载 ----------
spikes_nossdp = np.load("spikes_record.npy")
spikes_ssdp = np.load("spikes_record_SSDP.npy")

# ---------- reshape ----------
mean_ssdp = np.mean(spikes_ssdp, axis=0).T
mean_nossdp = np.mean(spikes_nossdp, axis=0).T

T = mean_ssdp.shape[1]
stage_percent = np.linspace(0, 120, T)
num_neurons = mean_ssdp.shape[0]

# ---------- 归一化 + 排序 ----------
def normalize_and_sort(spike_matrix):
    z_spikes = zscore(spike_matrix, axis=1)
    z_spikes[np.isnan(z_spikes)] = 0
    peak_indices = np.argmax(z_spikes, axis=1)
    sorted_idx = np.argsort(peak_indices)
    return z_spikes[sorted_idx], peak_indices[sorted_idx]

z_ssdp_sorted, peak_ssdp = normalize_and_sort(mean_ssdp)
z_nossdp_sorted, peak_nossdp = normalize_and_sort(mean_nossdp)

# ---------- 统计直方图 ----------
def histogram_peak_distribution(peak_array, num_bins=10):
    peak_stage_percent = peak_array / T * 100
    counts, bins = np.histogram(peak_stage_percent, bins=num_bins, range=(0, 120))
    return counts, bins

hist_ssdp, bins = histogram_peak_distribution(peak_ssdp)
hist_nossdp, _ = histogram_peak_distribution(peak_nossdp)

# ---------- 绘图 ----------
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5], hspace=0.4, wspace=0.3)

# A1
ax1 = fig.add_subplot(gs[0:2, 0])
sns.heatmap(z_ssdp_sorted, ax=ax1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
# ax1.set_title("Backprop + SSDP", fontsize=36,fontweight='bold')
ax1.set_ylabel("Neuron Index", fontsize=36,fontweight='bold')
ax1.set_xlabel("Stage (%)", fontsize=36,fontweight='bold')
ax1.set_xticks(np.linspace(0, T, 5))
ax1.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 120, 5)], fontsize=30,fontweight='bold')
ax1.tick_params(axis='y', labelsize=30)

# A2
ax2 = fig.add_subplot(gs[0:2, 1])
# 画热图，并保存 colorbar 引用
hm = sns.heatmap(z_nossdp_sorted, ax=ax2, cmap='jet', cbar=True,
                 cbar_kws={'label': 'z-score'},
                 xticklabels=False, yticklabels=False)

# 设置 colorbar 的 label 字体大小和刻度字体大小
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)         # 刻度字体大小
cbar.set_label("z-score", fontsize=30,fontweight='bold')    # 标签字体大小

# ax2.set_title("Backprop only", fontsize=36,fontweight='bold')
ax2.set_xlabel("Stage (%)", fontsize=36,fontweight='bold')
ax2.set_xticks(np.linspace(0, T, 5))
ax2.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 120, 5)], fontsize=30,fontweight='bold')


ax1.text(-0.11, 1.15, "a", transform=ax1.transAxes,
         fontsize=36, fontweight='bold', va='top', ha='left')

ax2.text(-0.11, 1.15, "b", transform=ax2.transAxes,
         fontsize=36, fontweight='bold', va='top', ha='left')



plt.tight_layout()
plt.savefig("final_layout_ssdp_population_zscore.pdf", dpi=300)
plt.show()
