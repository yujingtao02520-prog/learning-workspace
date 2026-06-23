# Text-to-Image (T2I) Diffusion Transformer & Flow Matching 技术原理

本项目实现了一个基于 **Diffusion Transformer (DiT)** 架构的文本引导图像生成 (Text-to-Image, T2I) 框架，并分别实现了 **Diffusion (DDPM)** 和 **Flow Matching (Rectified Flow)** 两套生成动力学方法作为对比。

---

## 1. 架构升级：从 U-Net 到 Diffusion Transformer (DiT)

传统的图像生成模型（如 DDPM 或 Stable Diffusion 1.x/2.x）大都采用以卷积为主的 **U-Net** 架构。而近年来以 Sora, Stable Diffusion 3, Flux, PixArt 为代表的先进生成模型全面转向了 **Transformer** 架构。

本项目设计并实现的 `DiT`（Diffusion Transformer）继承了 Vision Transformer (ViT) 的设计理念，其前向计算流水线如下：

1. **图像 Patch 化 (Patchify & Patch Embedding)**
   - 输入图像大小为 $H \times W \times C$（本项目默认单通道灰度图 $C=1$）。
   - 将图像切分为大小为 $P \times P$（如 $P=4$）的互不重叠块，共有 $N = \frac{H}{P} \times \frac{W}{P}$ 个 patch。
   - 通过卷积投影将每个 patch 映射到 $D$ 维特征空间，得到序列 $\mathbf{Z} \in \mathbb{R}^{N \times D}$。
   - 加入预先计算或可学习的 **2D 空间位置编码 (2D Position Embedding)**，赋予 Transformer 空间几何位置感知。

2. **自适应层归一化与时间调制 (AdaLN-Zero)**
   与标准 Transformer 的自适应层归一化相似，DiT 的自适应层归一化层 (Adaptive Layer Normalization, AdaLN) 利用时间步嵌入来调整归一化后的分布。在 AdaLN-Zero 机制中，我们在每一块的开始，将时间嵌入投影到缩放因子 $\gamma$、偏移因子 $\beta$ 和出门门限因子 $\alpha$：
   
   $$\text{AdaLN}(h, t_{\text{emb}}) = \gamma(t_{\text{emb}}) \cdot \text{LayerNorm}(h) + \beta(t_{\text{emb}})$$
   
   在初始化时，投影器的输出层权重与偏置全部设为 0。这意味着初始化时 $\gamma=0, \beta=0, \alpha=0$，残差网络中的 Transformer 块在初始状态下等价于恒等映射（Identity Mapping），这极大稳定了深层网络的初始化训练。

---

## 2. 文本条件注入：交叉注意力 (Cross-Attention)

为了让文本 Prompt 指导图像生成，模型引入了**交叉注意力机制 (Cross-Attention)**。

在每一层 DiTBlock 中：
- 设图像特征序列为 $\mathbf{X} \in \mathbb{R}^{N \times D}$
- 设经过文本编码器 (Text Encoder) 提取后的文本特征序列为 $\mathbf{C} \in \mathbb{R}^{L \times D}$（$L$ 为文本 Token 最大长度）

交叉注意力层以图像特征作为查询向量 (Query)，以文本特征作为键向量 (Key) 和值向量 (Value)：

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{C}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{C}\mathbf{W}_V$$

$$\text{Cross-Attention}(\mathbf{X}, \mathbf{C}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_k}}\right)\mathbf{V}$$

通过该公式，图像中的每一个 Patch 都可以根据其在生成过程中的需要，自适应地从文本序列中提取关键语义信息（例如“circle”或“digit nine”）。

---

## 3. 生成方法对比：Diffusion vs Flow Matching

本项目在同一套 DiT 骨干网络上实现了两种不同的生成路径学说：

### A. 条件 Diffusion / DDPM

DDPM 定义了一个离散的马尔可夫加噪过程，最终将图像变为纯高斯噪声。

- **前向加噪过程**：
  给定真实图像 $x_0$，在时间步 $t \in \{0, \dots, T-1\}$ 添加高斯噪声：
  
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
  
- **模型优化目标**：
  训练模型 $\epsilon_\theta(x_t, t, c)$ 去预测所加入的噪声 $\epsilon$：
  
  $$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, \epsilon, t, c} \left[ \| \epsilon_\theta(x_t, t, c) - \epsilon \|^2 \right]$$
  
- **反向采样过程**：
  从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步利用预测噪声去噪重构：
  
  $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

---

### B. 条件 Flow Matching (Rectified Flow)

Flow Matching 放弃了随机马尔可夫链的去噪形式，转而使用确定性常微分方程 (ODE) 的矢量流表示。

- **线性插值路径**：
  在连续时间 $t \in [0, 1]$ 上，直接在噪声 $x_0 \sim \mathcal{N}(0, I)$ 和图像数据 $x_1 \sim p_{\text{data}}$ 之间拉出一条直线段：
  
  $$x_t = (1 - t)x_0 + t x_1$$
  
- **目标速度场 (Target Velocity Field)**：
  这条路径关于时间 $t$ 的一阶导数（即粒子移动的速度）是恒定的：
  
  $$u_t = \frac{dx_t}{dt} = x_1 - x_0$$
  
- **模型优化目标**：
  训练网络 $v_\theta(x_t, t, c)$ 拟合在给定条件 $c$ 下该流速场 $u_t$：
  
  $$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, x_1, t, c} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]$$
  
- **反向采样过程**：
  采样是一个解初值 ODE 的问题。从 $x(0) \sim \mathcal{N}(0, I)$ 开始，按照模型预测的速度向前流动积分到 $t=1$。
  
  - **Euler 积分器**：
    
    $$x_{k+1} = x_k + v_\theta(x_k, t_k, c) \cdot \Delta t$$
    
  - **Heun 积分器 (预估-校正器)**：
    先用 Euler 估算下一步位置 $\tilde{x}_{k+1}$，再取两点速度的平均值进行更新：
    
    $$\tilde{x}_{k+1} = x_k + v_k \cdot \Delta t$$
    
    $$x_{k+1} = x_k + \frac{1}{2} (v_k + v_\theta(\tilde{x}_{k+1}, t_{k+1}, c)) \cdot \Delta t$$

---

## 4. 无分类器引导 (Classifier-Free Guidance, CFG)

无论在 Diffusion 还是 Flow Matching 中，单纯依靠文本条件往往会导致生成的图像语义漂移或不够清晰。**Classifier-Free Guidance (CFG)** 是一种在生成采样阶段增强文本对齐度和图像对比度的关键技术。

### 训练阶段
在训练过程中，模型以一定的概率（如 $10\%$）随机将文本 Token 全部替换成空置占位符（即无条件 `<pad>` 标记，对应 ID 为 0）。这样模型就能同时学会**无条件生成模型**和**有条件生成模型**。

### 采样阶段
采样时，我们在每一步同时预测有条件和无条件下的网络输出，并做插值外推。

- **Diffusion 中的 CFG**：
  调整后的噪声预测为：
  
  $$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot ( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset) )$$
  
- **Flow Matching 中的 CFG**：
  调整后的速度场预测为：
  
  $$\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, \emptyset) + s \cdot ( v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset) )$$

其中 $s \ge 1.0$ 是引导尺度参数 (CFG Scale)。通过增大 $s$，可以让模型生成的图像更紧密地匹配文本 Prompt（提升图像饱和度与对比度，减少模糊），代价是生成多样性的轻微降低。
