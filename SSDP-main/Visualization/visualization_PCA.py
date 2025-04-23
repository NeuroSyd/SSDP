import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载 spike 数据
spikes_withSSDP = np.load("spikes_record_SSDP.npy")      # shape: [N, T, C]
spikes_withoutSSDP = np.load("spikes_record.npy")        # shape: [N, T, C]

# 特征提取（时间维度 mean pooling）
features_ssdp = spikes_withSSDP.mean(axis=1)
features_nossdp = spikes_withoutSSDP.mean(axis=1)

# 拼接特征和标签
X = np.concatenate([features_ssdp, features_nossdp], axis=0)
group = np.array(['SSDP'] * len(features_ssdp) + ['Without SSDP'] * len(features_nossdp))

# 标准化
X_std = StandardScaler().fit_transform(X)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# 可视化
plt.figure(figsize=(14, 8))
for g, color in zip(['Without SSDP', 'SSDP'], ['steelblue', 'orange']):
    idx = group == g
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=g, alpha=0.6, s=100, c=color)

plt.xlabel("PC1", fontsize=36,fontweight='bold')
plt.ylabel("PC2", fontsize=36,fontweight='bold')
plt.xticks(fontsize=30,fontweight='bold')
plt.yticks(fontsize=30,fontweight='bold')
plt.legend(fontsize=30, loc='best',prop={'size': 30, 'weight': 'bold'})

# === 添加粗体图标 a ===
plt.text(-0.13, 1.08, 'b', transform=plt.gca().transAxes,
         fontsize=36, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.show()
