import numpy as np
import matplotlib.pyplot as plt

def get_rope_frequencies(dim, base=10000):
    """
    计算 RoPE 的旋转频率 theta_i
    每个 2D 子空间对应一个频率
    """
    assert dim % 2 == 0, "维度必须是偶数"
    # 计算 theta_i = base^(-2i / dim)
    # 共有 dim // 2 个频率
    theta = 1.0 / (base ** (np.arange(0, dim, 2)[: (dim // 2)].astype(float) / dim))
    return theta

def apply_rope_2d(x, position, theta):
    """
    在单个 2D 子空间内，将向量旋转 position * theta 角度
    [x0, x1] 旋转矩阵乘法
    """
    angle = position * theta
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    
    # 旋转矩阵 R = [[cos, -sin], [sin, cos]]
    # R * x
    x_rotated = np.array([
        x[0] * cos_val - x[1] * sin_val,
        x[0] * sin_val + x[1] * cos_val
    ])
    return x_rotated

def apply_rope_full(x, position, frequencies):
    """
    对高维向量 x 应用完整的 RoPE 位置编码
    把向量切分成若干个 2D 平面，分别在对应的平面进行旋转
    """
    dim = len(x)
    x_rotated = np.zeros_like(x)
    
    for i in range(dim // 2):
        # 提取第 i 个 2D 平面的分量
        x_2d = x[2*i : 2*i + 2]
        # 获取对应的旋转频率
        theta_i = frequencies[i]
        # 进行 2D 旋转
        x_rotated_2d = apply_rope_2d(x_2d, position, theta_i)
        # 存回结果
        x_rotated[2*i : 2*i + 2] = x_rotated_2d
        
    return x_rotated

# =====================================================================
# 主程序：验证 RoPE 的相对距离注意力衰减特性
# =====================================================================
# 1. 定义向量维度
dim = 64
frequencies = get_rope_frequencies(dim, base=10000)

# 2. 随机生成一个 Query 向量 q 和 Key 向量 k
np.random.seed(42)
q = np.random.randn(dim)
k = np.random.randn(dim)
# 归一化，使得初始内积可控
q /= np.linalg.norm(q)
k /= np.linalg.norm(k)

print("=== 旋转位置编码 (RoPE) 实验 ===")
print(f"向量维度: {dim}，共有 {dim//2} 个独立旋转的 2D 平面")
print(f"位置 0 时，未旋转的原始 q 与 k 内积 (点积): {np.dot(q, k):.4f}")

# 3. 固定 Query 在位置 0，让 Key 的位置从 0 移动到 100，观察内积如何随相对距离变化
q_pos = 0
distances = np.arange(0, 100)
dot_products = []

# 对 Query 编码 (在位置 0)
q_rotated = apply_rope_full(q, q_pos, frequencies)

for dist in distances:
    k_pos = dist
    # 对 Key 编码 (在位置 dist)
    k_rotated = apply_rope_full(k, k_pos, frequencies)
    
    # 计算旋转后的内积
    dot_val = np.dot(q_rotated, k_rotated)
    dot_products.append(dot_val)

# =====================================================================
# 4. 可视化：绘制内积随着相对距离衰减的曲线
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(distances, dot_products, 'b-', linewidth=2, label='RoPE Dot Product')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('RoPE Decay Effect: Query at Pos 0, Key at Pos [0-100]', fontsize=14)
plt.xlabel('Relative Distance (Key Pos - Query Pos)', fontsize=12)
plt.ylabel('Attention Score (Dot Product of Rotated Vectors)', fontsize=12)
plt.grid(True, alpha=0.3)

# 标出一些重点
plt.annotate('Max Similarity\n(At same position)', xy=(0, dot_products[0]), xytext=(15, dot_products[0]-0.15),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

# 说明文字：为什么会有这种衰减？
# 因为不同频率的 2D 旋转相叠加，会在距离增加时产生“干涉消振”的效应，使得总内积趋于 0。
# 这给大模型提供了天然的“距离偏置”：距离越近，注意力越强。
plt.text(40, max(dot_products)*0.6, 
         "Why it decays?\n"
         "Integrating rotation matrices across\n"
         "multiple frequencies acts as a band-pass filter.\n"
         "As distance increases, the phase differences\n"
         "cause destructive interference, naturally\n"
         "decaying the inner product.",
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8), fontsize=10)

plt.legend()
print("\n[提示] RoPE 相对位置衰减图表已生成，请在本地运行查看！")
plt.show()
