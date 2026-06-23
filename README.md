# 线性代数与空间变换在 AI 中的运用：从几何直觉到算法实践

欢迎开启这段学习旅程！本教程的核心理念是：**用“空间几何变换”的直觉去理解抽象的线性代数公式，并直接建立其与人工智能（AI）主流算法的关联。**

线性代数不仅是数学符号，更是对“空间”进行旋转、拉伸、切变和投影的科学。在现代 AI 中，从神经网络的权重矩阵，到大模型的注意力机制，本质上都是高维特征空间中的几何变换。

---

## 🗺️ 学习路线图 (Curriculum Map)

本学习路径共分为 5 个阶段，每个阶段都将包含**数学几何本质**与**AI 算法映射**：

```
[01_Vectors_and_Spaces] ─────► [02_Linear_Transformations] ─────► [03_Matrix_Inverse_and_Systems]
   (向量、基底与高维嵌入)             (矩阵映射、行列式与线性层)           (逆变换、投影与最小二乘回归)
                                                                               │
                                                                               ▼
[05_Special_Transformations] ◄───────────────────────────────────── [04_Eigenvalues_and_SVD]
   (仿射变换、核函数与 RoPE 位置编码)                                     (特征值、SVD 与 LoRA 低秩微调)
```

| 章节目录 | 核心数学概念 | 对应的 AI 应用场景 |
| :--- | :--- | :--- |
| **[01 向量与空间基础](./01_Vectors_and_Spaces/)** | 线性组合、张成空间 (Span)、基底 (Basis)、内积 | 词嵌入 (Embedding)、语义搜索中的余弦相似度 |
| **[02 线性变换与矩阵](./02_Linear_Transformations/)** | 空间变换、矩阵乘法（复合变换）、行列式（体积缩放） | 神经网络 Linear 层、Transformer 特征映射 (QKV) |
| **[03 逆变换与线性方程组](./03_Matrix_Inverse_and_Systems/)** | 逆矩阵、秩 (Rank)、列空间、正交投影 | 线性回归（最小二乘法几何解释）、AE信息重构 |
| **[04 特征值与 SVD](./04_Eigenvalues_and_SVD/)** | 特征值/特征向量、SVD 空间重构、低秩近似 | PCA 主成分分析降维、大模型低秩微调 (LoRA) |
| **[05 特征空间变换进阶](./05_Special_Transformations_in_AI/)** | 仿射变换、齐次坐标、正交旋转、核映射 | 图像数据增强、SVM 核方法、大模型位置编码 (RoPE) |

---

## 🛠️ 环境准备与学习建议

### 1. 推荐学习资源
- **视频课程**：强烈推荐 3Blue1Brown 的视频系列 [《线性代数的本质》](https://space.bilibili.com/884616/channel/detail?cid=9447)。本教程的几何视角与其高度契合。
- **经典教材**：Gilbert Strang 教授的 *Introduction to Linear Algebra*（《引入线性代数》）。

### 2. 实践环境搭建
在每个章节的 `scratch/` 目录下，我们为你准备了动手实践的 Python 文件。为了运行这些代码，建议在本地安装以下 Python 库：

```bash
pip install numpy matplotlib torch
```
- **NumPy**：用于基础的矩阵运算和线性代数求解。
- **Matplotlib**：用于绘制向量和空间变换的几何动画与图表。
- **PyTorch**：用于构建简单的神经网络，感受张量（Tensor）运算 and 空间变换。

---

## ✍️ 如何开始？
直接进入 **[01_Vectors_and_Spaces](./01_Vectors_and_Spaces/README.md)** 目录，阅读对应的学习指南，并尝试运行 `scratch/` 下的示例代码吧！
