import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(matrix, title="Linear Transformation", shear_val=0):
    """
    绘制线性变换前后网格的变化
    """
    # 1. 产生一组 2D 网格点
    x = np.linspace(-2, 2, 11)
    y = np.linspace(-2, 2, 11)
    X, Y = np.meshgrid(x, y)
    
    # 展平成 (2, N) 形状的矩阵，方便进行矩阵乘法
    points = np.vstack([X.flatten(), Y.flatten()])
    
    # 2. 应用矩阵变换: Y = A * X
    transformed_points = np.dot(matrix, points)
    
    # 重新塑形成网格形状以便绘图
    TX = transformed_points[0, :].reshape(X.shape)
    TY = transformed_points[1, :].reshape(Y.shape)
    
    # 计算行列式 (面积缩放因子)
    det = np.linalg.det(matrix)
    
    # 3. 开始绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制原始空间
    ax_orig = axes[0]
    ax_orig.set_title("Original Grid (Det = 1.0)", fontsize=12)
    # 画网格线
    for i in range(X.shape[0]):
        ax_orig.plot(X[i, :], Y[i, :], color='lightblue', alpha=0.5)
        ax_orig.plot(X[:, i], Y[:, i], color='lightblue', alpha=0.5)
    # 画基向量
    ax_orig.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='red', width=0.015, label='i_hat [1, 0]')
    ax_orig.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='green', width=0.015, label='j_hat [0, 1]')
    ax_orig.set_xlim(-4, 4)
    ax_orig.set_ylim(-4, 4)
    ax_orig.grid(True, alpha=0.3)
    ax_orig.axhline(0, color='black', linewidth=0.8)
    ax_orig.axvline(0, color='black', linewidth=0.8)
    ax_orig.legend()
    
    # 绘制变换后的空间
    ax_trans = axes[1]
    ax_trans.set_title(f"{title} (Det = {det:.2f})", fontsize=12)
    # 画变换后的网格线
    for i in range(TX.shape[0]):
        ax_trans.plot(TX[i, :], TY[i, :], color='pink', alpha=0.6)
        ax_trans.plot(TX[:, i], TY[:, i], color='pink', alpha=0.6)
        
    # 计算基向量变换后的目的地
    i_new = np.dot(matrix, np.array([1, 0]))
    j_new = np.dot(matrix, np.array([0, 1]))
    
    # 画变换后的基向量
    ax_trans.quiver(0, 0, i_new[0], i_new[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.015, label=f'i_new {i_new}')
    ax_trans.quiver(0, 0, j_new[0], j_new[1], angles='xy', scale_units='xy', scale=1, color='green', width=0.015, label=f'j_new {j_new}')
    ax_trans.set_xlim(-4, 4)
    ax_trans.set_ylim(-4, 4)
    ax_trans.grid(True, alpha=0.3)
    ax_trans.axhline(0, color='black', linewidth=0.8)
    ax_trans.axvline(0, color='black', linewidth=0.8)
    ax_trans.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 演示 1: 缩放变换矩阵 (Scaling Matrix)
# ==========================================
# X方向拉伸2倍，Y方向压缩0.5倍
scale_matrix = np.array([
    [2.0, 0.0],
    [0.0, 0.5]
])
print("--- 1. 缩放变换矩阵 ---")
print(scale_matrix)
print("基向量 i_hat 变成了:", scale_matrix[:, 0])
print("基向量 j_hat 变成了:", scale_matrix[:, 1])
# plot_transformation(scale_matrix, "Scaling")

# ==========================================
# 演示 2: 旋转变换矩阵 (Rotation Matrix)
# ==========================================
# 逆时针旋转 45 度
theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array([
    [c, -s],
    [s,  c]
])
print("\n--- 2. 旋转变换矩阵 (45度) ---")
print(rotation_matrix)
# plot_transformation(rotation_matrix, "Rotation (45 deg)")

# ==========================================
# 演示 3: 切变变换矩阵 (Shear Matrix)
# ==========================================
# Y坐标保持不变，X坐标加上 1.5 * Y 坐标 (拉斜)
shear_matrix = np.array([
    [1.0, 1.5],
    [0.0, 1.0]
])
print("\n--- 3. 切变变换矩阵 (X方向拉斜) ---")
print(shear_matrix)
# plot_transformation(shear_matrix, "Shear Transformation")

# ==========================================
# 演示 4: 降维变换（奇异矩阵，行列式为 0）
# ==========================================
# 把整个 2D 空间压缩到 y = x 的一条直线上
singular_matrix = np.array([
    [1.0, 1.0],
    [1.0, 1.0]
])
print("\n--- 4. 降维挤压变换 (行列式为0) ---")
print(singular_matrix)
print(f"行列式计算值: {np.linalg.det(singular_matrix)}")

# 执行画图，请取消下面你想查看的变换注释：
plot_transformation(shear_matrix, "Shear Transformation")
# plot_transformation(rotation_matrix, "Rotation (45 deg)")
# plot_transformation(singular_matrix, "Singular (Det = 0)")
