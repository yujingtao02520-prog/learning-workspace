import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================================
# 1. 准备数据并使用正规方程 (Normal Equation) 求解最小二乘回归
# 我们有三个样本点 (x, y): (1, 2), (2, 2), (3, 4)
# 拟合模型: y = m * x (为了直观，我们省去截距，使列空间为 3D 中的一条线)
# =====================================================================
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 2.0, 4.0])

# 构建设计矩阵 A (列向量) 和 观测向量 b
A = x_data.reshape(-1, 1) # 形状 (3, 1), 代表 A 的列空间是一条 3D 直线
b = y_data                 # 形状 (3,), 代表我们的目标向量

print("=== 最小二乘法的空间几何投影 ===")
print("设计矩阵 A (即特征列向量):\n", A)
print("观测目标向量 b:\n", b)

# 计算正规方程: A^T * A * m = A^T * b  =>  m = (A^T * A)^-1 * A^T * b
AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)
m = np.linalg.inv(AtA).dot(Atb)[0]
print(f"\n拟合斜率 m = {m:.4f}")
print(f"拟合方程为: y = {m:.4f} * x")

# 计算列空间中的投影向量 p = A * m
p = A.flatten() * m
# 计算误差向量 (残差) e = b - p
e = b - p

print(f"投影向量 p (列空间中距离 b 最近的向量): {p}")
print(f"误差向量 e (b 与投影点的偏差): {e}")
# 几何验证：误差向量 e 应当与列空间（即列向量 A）正交，即内积为 0
dot_product = np.dot(A.flatten(), e)
print(f"验证正交性 (A . e): {dot_product:.6f} (极其接近 0，说明垂直！)")


# =====================================================================
# 2. 3D 可视化：观察向量 b 投影到 A 的列空间
# 在这个绘图里，3D 空间的三个轴分别代表 3 个样本点的值。
# =====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画原点到目标向量 b 的箭头 (红色)
ax.quiver(0, 0, 0, b[0], b[1], b[2], color='red', arrow_length_ratio=0.1, label='Target Vector b (Actual Data)', linewidth=2)

# 画列空间 A (因为只有一个列向量，其张成空间是一条过原点的无限延长线)
# 我们用虚线画出这条线
line_t = np.linspace(-0.5, 1.5, 100)
line_points = np.outer(line_t, A.flatten() * 1.2)
ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'g--', label='Column Space of A (Model Span)')

# 画投影向量 p 的箭头 (蓝色)
ax.quiver(0, 0, 0, p[0], p[1], p[2], color='blue', arrow_length_ratio=0.1, label='Projection p (Best Prediction)', linewidth=2)

# 画误差向量 e (从投影点 p 指向 目标点 b)
ax.quiver(p[0], p[1], p[2], e[0], e[1], e[2], color='orange', arrow_length_ratio=0.1, label='Error Vector e (Residual)', linestyle=':')

# 标出点的位置
ax.scatter(b[0], b[1], b[2], color='red', s=50)
ax.scatter(p[0], p[1], p[2], color='blue', s=50)

# 设置轴标签
# 轴 1 代表第一个点 x=1 时的观测值，以此类推
ax.set_xlabel('Dimension 1 (Point 1, x=1)')
ax.set_ylabel('Dimension 2 (Point 2, x=2)')
ax.set_zlabel('Dimension 3 (Point 3, x=3)')
ax.set_title('Geometric View of Least Squares Regression (Projection in 3D)')

# 优化视角和显示
ax.view_init(elev=20, azim=45)
ax.grid(True)
ax.legend()

# =====================================================================
# 3. 2D 常规数据拟合图（传统视角对比）
# =====================================================================
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, color='red', zorder=5, label='Data Points')
x_fit = np.linspace(0, 4, 100)
y_fit = m * x_fit
plt.plot(x_fit, y_fit, 'b-', label=f'Fitted line (y = {m:.2f}x)')
for xi, yi in zip(x_data, y_data):
    plt.plot([xi, xi], [yi, m*xi], 'orange', linestyle=':') # 画残差线
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traditional View of Fitting')
plt.legend()
plt.grid(True, alpha=0.3)

print("\n[提示] 3D 与 2D 绘图窗口已生成，请在本地运行查看！")
plt.show()
