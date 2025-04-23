import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances

# === 加载数据 ===
spikes_withSSDP = np.load("spikes_record_SSDP.npy")      # shape: [N, T, C]
spikes_withoutSSDP = np.load("spikes_record.npy")         # shape: [N, T, C]

# === 加载或创建 labels ===
try:
    labels = np.load("labels.npy")
    has_labels = True
except FileNotFoundError:
    labels = None
    has_labels = False
    print("未找到 labels.npy，使用 SSDP/Without SSDP 分组标签")

# === 特征提取 ===
def extract_features(spike_tensor, mode='mean'):
    if mode == 'mean':
        return spike_tensor.mean(axis=1)
    elif mode == 'sum':
        return spike_tensor.sum(axis=1)

features_ssdp = extract_features(spikes_withSSDP, mode='mean')
features_nossdp = extract_features(spikes_withoutSSDP, mode='mean')

X = np.concatenate([features_ssdp, features_nossdp], axis=0)
group = np.array(['SSDP'] * len(features_ssdp) + ['Without SSDP'] * len(features_nossdp))  # 更清晰的标签

if has_labels:
    y = np.concatenate([labels, labels])
else:
    y = group  # 用于 silhouette 评分

# === 标准化 & t-SNE ===
X_std = StandardScaler().fit_transform(X)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_std)

# === 定量指标：类内 & 类间距离 ===
def average_intra_distance(features):
    dists = pairwise_distances(features)
    n = len(features)
    return dists[np.triu_indices(n, k=1)].mean()

intra_ssdp = average_intra_distance(features_ssdp)
intra_nossdp = average_intra_distance(features_nossdp)
inter_distance = np.linalg.norm(features_ssdp.mean(axis=0) - features_nossdp.mean(axis=0))

print("\n=== Quantitative Analysis ===")
print(f"Average Intra-cluster Distance (SSDP):         {intra_ssdp:.4f}")
print(f"Average Intra-cluster Distance (Without SSDP): {intra_nossdp:.4f}")
print(f"Inter-cluster Distance (between SSDP and Without SSDP): {inter_distance:.4f}")

# === silhouette score（整组可分性）===
sil_score = silhouette_score(X_embedded, y)
print(f"Silhouette Score (t-SNE space): {sil_score:.4f}")

# === 绘图 ===
colors = {
    'Without SSDP': 'steelblue',
    'SSDP': 'orange'
}

plt.figure(figsize=(14, 8))
for g in ['Without SSDP', 'SSDP']:
    idx = group == g
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                label=g,
                alpha=0.6,
                s=100,
                c=colors[g])

plt.xlabel("t-SNE dim 1", fontsize=36,fontweight='bold')
plt.ylabel("t-SNE dim 2", fontsize=36,fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

# legend 顺序固定
handles, labels_ = plt.gca().get_legend_handles_labels()
desired_order = ['Without SSDP', 'SSDP']
sorted_handles_labels = sorted(zip(handles, labels_), key=lambda x: desired_order.index(x[1]))
handles_sorted, labels_sorted = zip(*sorted_handles_labels)
plt.legend(handles_sorted, labels_sorted, loc='best', fontsize=30, prop={'size': 30, 'weight': 'bold'})

# === 添加粗体图标 a ===
plt.text(-0.13, 1.08, 'a', transform=plt.gca().transAxes,
         fontsize=36, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.show()
