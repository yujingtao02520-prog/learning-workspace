import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. 生成一个合成的结构化矩阵 (模拟一张灰度图像)
# 该图像包含明显的垂直和水平线条，有利于演示低秩近似的效果
# =====================================================================
# 创建一个 100x100 的网格
size = 100
image = np.zeros((size, size))

# 绘制一些图案 (水平和垂直条纹、中间的实心方块)
# 这会让矩阵的某些奇异值非常大，而大多数非常小
image[20:80, 20:80] = 0.8
image[35:65, 35:65] = 0.2
for i in range(size):
    image[i, i] += 0.15 # 添加对角线线索
    image[i, size - 1 - i] += 0.15

print("=== 图像矩阵 SVD 分解 ===")
print(f"图像尺寸: {image.shape}")

# =====================================================================
# 2. 对图像进行奇异值分解 (SVD): A = U * Sigma * V^T
# =====================================================================
U, s, Vt = np.linalg.svd(image)

print(f"左奇异矩阵 U 形状: {U.shape}")
print(f"奇异值向量 s 长度 (共有 {len(s)} 个奇异值): {s.shape}")
print(f"右奇异矩阵 V^T 形状: {Vt.shape}")

# 计算奇异值的能量分布（前k个奇异值占总和的比例）
cumulative_energy = np.cumsum(s) / np.sum(s)
print(f"前 1 个奇异值包含能量: {cumulative_energy[0]*100:.2f}%")
print(f"前 5 个奇异值包含能量: {cumulative_energy[4]*100:.2f}%")
print(f"前 15 个奇异值包含能量: {cumulative_energy[14]*100:.2f}%")

# =====================================================================
# 3. 低秩近似重构
# 定义函数，只保留前 k 个奇异值来重构图像
# =====================================================================
def reconstruct_low_rank(U, s, Vt, k):
    """
    使用前 k 个奇异值重构矩阵
    A_k = U[:, :k] * diag(s[:k]) * V^T[:k, :]
    """
    # 提取前 k 个分量
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    
    # 重构矩阵
    reconstructed = np.dot(U_k, np.dot(s_k, Vt_k))
    return reconstructed

# 选取不同的秩 k 进行重构
ranks = [1, 3, 10, 30]
reconstructed_images = [reconstruct_low_rank(U, s, Vt, r) for r in ranks]

# =====================================================================
# 4. 可视化：绘制原图与各种低秩近似的对比
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 绘制原始图像
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original Image (Rank 100)")
axes[0, 0].axis('off')

# 绘制各个秩的重构图像
for idx, r in enumerate(ranks):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(reconstructed_images[idx], cmap='gray')
    axes[row, col].set_title(f"Rank {r} Approximation\n(Energy: {cumulative_energy[r-1]*100:.1f}%)")
    axes[row, col].axis('off')

# 绘制奇异值下降曲线
ax_curve = axes[1, 2]
ax_curve.plot(s, 'r-', linewidth=2, label='Singular Values')
ax_curve.set_title("Singular Value Decay")
ax_curve.set_xlabel("Index")
ax_curve.set_ylabel("Value")
ax_curve.grid(True, alpha=0.3)

# 绘制累积能量图 (在同一个图表上用双 Y 轴表示)
ax_energy = ax_curve.twinx()
ax_energy.plot(cumulative_energy, 'b--', linewidth=1.5, label='Cumulative Energy')
ax_energy.set_ylabel("Cumulative Energy Ratio")
ax_energy.legend(loc='lower right')

plt.tight_layout()
print("\n[提示] 重构对比与奇异值分析图表已生成，请在本地运行查看！")
plt.show()
