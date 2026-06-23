import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================================
# 1. 初始化画布与子图
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Linear Algebra Basics: Vector Operations Animation", fontsize=16, fontweight='bold', color='#1a1a1a')

# 定义向量数据
u = np.array([3.0, 1.0])
v = np.array([1.0, 3.0])
w = u + v  # [4, 4]

# 设置子图 1 (向量加法)
ax1.set_xlim(-1, 6)
ax1.set_ylim(-1, 6)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)
ax1.set_title("Vector Addition: u + v", fontsize=13, fontweight='bold')
ax1.set_aspect('equal')

# 设置子图 2 (数乘向量)
ax2.set_xlim(-4, 6)
ax2.set_ylim(-4, 6)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_title("Scalar Multiplication: c * v", fontsize=13, fontweight='bold')
ax2.set_aspect('equal')

# =====================================================================
# 2. 定义绘图元素 (Arrows and Texts)
# =====================================================================
# 子图 1 元素
quiver_u = ax1.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#FF5733', width=0.015, label='Vector u = [3, 1]')
quiver_v_shifted = ax1.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#33FF57', width=0.015, label='Vector v = [1, 3] (Shifted)')
quiver_w = ax1.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#3357FF', width=0.018, label='u + v = [4, 4]')
text_u = ax1.text(0, 0, '', fontsize=11, color='#FF5733', fontweight='bold')
text_v = ax1.text(0, 0, '', fontsize=11, color='#27ae60', fontweight='bold')
text_w = ax1.text(0, 0, '', fontsize=11, color='#3357FF', fontweight='bold')
ax1.legend(loc='upper left')

# 子图 2 元素
v_base = np.array([2.0, 1.5])
quiver_v_base = ax2.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#8e44ad', width=0.015, label='Base Vector v = [2, 1.5]')
quiver_scaled = ax2.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#f1c40f', width=0.012, label='c * v (Scaled)')
text_scaled = ax2.text(0, 0, '', fontsize=12, color='#d35400', fontweight='bold')
ax2.legend(loc='upper left')

# =====================================================================
# 3. 动画更新函数
# 动画帧总数：200 帧
# =====================================================================
def update(frame):
    # ------------------
    # 左侧：向量加法动画
    # ------------------
    # 0 - 40 帧：绘制向量 u [3, 1] 逐渐增长
    if frame <= 40:
        ratio = frame / 40.0
        curr_u = u * ratio
        quiver_u.set_UVC(curr_u[0], curr_u[1])
        text_u.set_text(f"u [{curr_u[0]:.1f}, {curr_u[1]:.1f}]")
        text_u.set_position((curr_u[0]/2, curr_u[1]/2 - 0.3))
        
        # 隐藏 v 和 w
        quiver_v_shifted.set_UVC(0, 0)
        quiver_w.set_UVC(0, 0)
        text_v.set_text('')
        text_w.set_text('')
        
    # 41 - 90 帧：绘制向量 v 平移到 u 的终点，并逐渐增长
    elif frame <= 90:
        ratio = (frame - 40) / 50.0
        curr_v = v * ratio
        
        # u 保持完整
        quiver_u.set_UVC(u[0], u[1])
        # v 从 u 的终点出发
        quiver_v_shifted.set_offsets(u)
        quiver_v_shifted.set_UVC(curr_v[0], curr_v[1])
        
        text_v.set_text(f"v [{curr_v[0]:.1f}, {curr_v[1]:.1f}]")
        text_v.set_position((u[0] + curr_v[0]/2 + 0.2, u[1] + curr_v[1]/2))
        
        # 隐藏 w
        quiver_w.set_UVC(0, 0)
        text_w.set_text('')
        
    # 91 - 140 帧：绘制从原点到 v 终点的和向量 w 逐渐增长
    elif frame <= 140:
        ratio = (frame - 91) / 49.0
        curr_w = w * ratio
        
        # u 和 v 保持完整
        quiver_u.set_UVC(u[0], u[1])
        quiver_v_shifted.set_offsets(u)
        quiver_v_shifted.set_UVC(v[0], v[1])
        
        # w 增长
        quiver_w.set_UVC(curr_w[0], curr_w[1])
        
        text_w.set_text(f"u+v [{curr_w[0]:.1f}, {curr_w[1]:.1f}]")
        text_w.set_position((curr_w[0]/2 - 0.5, curr_w[1]/2 + 0.3))
        
    # 141 - 200 帧：保持加法静止，展示结果
    else:
        quiver_u.set_UVC(u[0], u[1])
        quiver_v_shifted.set_offsets(u)
        quiver_v_shifted.set_UVC(v[0], v[1])
        quiver_w.set_UVC(w[0], w[1])

    # ------------------
    # 右侧：向量数乘动画 (c * v)
    # 标量 c 从 0 -> 2 -> -1.5 -> 1 变化
    # ------------------
    # 我们可以用三角函数平滑地生成 c 的循环值
    # 0 -> 2 -> -1.5 -> 1
    if frame <= 50:
        # 0 帧到 50 帧，c 从 1 变化到 2
        c = 1.0 + (frame / 50.0) * 1.0
    elif frame <= 110:
        # 50 帧到 110 帧，c 从 2 变化到 -1.5
        c = 2.0 - ((frame - 50) / 60.0) * 3.5
    elif frame <= 170:
        # 110 帧到 170 帧，c 从 -1.5 变化到 1
        c = -1.5 + ((frame - 110) / 60.0) * 2.5
    else:
        # 170 帧到 200 帧，保持 c = 1.0
        c = 1.0
        
    # 绘制基础向量 v
    quiver_v_base.set_UVC(v_base[0], v_base[1])
    
    # 绘制缩放向量 c*v
    v_scaled = c * v_base
    quiver_scaled.set_UVC(v_scaled[0], v_scaled[1])
    
    text_scaled.set_text(f"c = {c:.2f}\nc * v = [{v_scaled[0]:.2f}, {v_scaled[1]:.2f}]")
    # 把文本放在轴上方
    text_scaled.set_position((-3.5, 4.0))

    return quiver_u, quiver_v_shifted, quiver_w, text_u, text_v, text_w, quiver_v_base, quiver_scaled, text_scaled

# 创建动画 (每帧 50 毫秒，共 200 帧，循环播放)
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.tight_layout()
print("=== 动画创建成功 ===")
print("请在本地终端运行 python vector_basics_animation.py 查看交互式动画窗口。")
plt.show()
