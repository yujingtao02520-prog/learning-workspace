# Flow Matching 与 Diffusion/DDPM 图像生成 Demo

这是一个小型 PyTorch 图像生成项目，用同一个轻量级 `TinyUNet` 对比两类生成方法：

- **Diffusion / DDPM**：训练模型从带噪图像中预测噪声，生成时从高斯噪声逐步去噪。
- **Flow Matching / Rectified Flow**：训练模型学习速度场，生成时通过 ODE 积分把噪声样本流动到图像样本。

项目默认已经升级到 `64 x 64` 单通道图像和更深的 U-Net；仍然可以通过参数切回 `28 x 28` 轻量 demo。

## 快速开始

如果你只想验证脚本能跑通：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 2
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt

python train_flow_matching.py --dataset synthetic_shapes --epochs 2
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt
```

如果你有高端 GPU（如 RTX 5090）并想跑出更好的效果：

```bash
python train_diffusion.py --dataset fashion_mnist --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --num_timesteps 500 --sample_every 2
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt --num_samples 64

python train_flow_matching.py --dataset fashion_mnist --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --sample_steps 100 --sample_every 2
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt --num_samples 64 --sample_steps 100
```

## 环境安装

```bash
pip install -r requirements.txt
```

依赖保持最小化，只使用 `torch`、`torchvision`、`numpy`、`matplotlib`、`tqdm` 和 `pillow`，不依赖 Lightning、Hydra、wandb 或 diffusers。

## 数据集说明

本项目支持三种数据集：

- `mnist`：默认选项，使用 `torchvision.datasets.MNIST`。
- `fashion_mnist`：使用 `torchvision.datasets.FashionMNIST`。
- `synthetic_shapes`：无需联网，自动生成圆形、方形、三角形等简单二值图像，适合作为离线验收数据集。

所有图像都会被转换为 `[B, 1, H, W]`，默认 `H=W=64`，像素范围归一化到 `[-1, 1]`。

## 图像生成问题的基本形式

一张图像可以看成高维向量：

$$
x \in \mathbb{R}^{C \times H \times W}
$$

生成模型的目标，是从一个容易采样的简单分布生成真实数据分布：

$$
p_0 \rightarrow p_{\text{data}}
$$

其中 \(p_0\) 通常是标准高斯噪声分布：

$$
x_0 \sim \mathcal{N}(0, I)
$$

## Diffusion / DDPM 原理

DDPM 先定义一个固定的前向加噪过程，把真实图像逐步变成噪声。给定真实图像 \(x_0\)，第 \(t\) 步的条件分布为：

$$
q(x_t \mid x_0) = \mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

其中：

$$
\alpha_t = 1 - \beta_t,\qquad
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

等价的重参数化采样公式为：

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

模型学习预测加入的噪声：

$$
\epsilon_\theta(x_t, t)
$$

训练损失为：

$$
\mathcal{L}_{\text{DDPM}}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
\epsilon_\theta(x_t,t)-\epsilon
\right\|_2^2
\right]
$$

采样时从纯噪声 \(x_T \sim \mathcal{N}(0,I)\) 开始，逐步近似反向分布：

$$
p_\theta(x_{t-1}\mid x_t)
$$

常用更新公式为：

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z
$$

其中 \(z \sim \mathcal{N}(0,I)\)，最后一步通常不再加入噪声。

## Flow Matching / Rectified Flow 原理

Flow Matching 学习一个时间相关的速度场：

$$
v_\theta(x_t,t)
$$

这个速度场描述样本在时间 \(t\) 应该如何移动，使其从噪声分布流向真实数据分布。生成过程可写成 ODE：

$$
\frac{dx}{dt} = v_\theta(x,t)
$$

训练时，从噪声样本和真实样本之间构造一条线性路径：

$$
x_t = (1-t)x_0 + tx_1
$$

其中：

$$
x_0 \sim \mathcal{N}(0,I),\qquad
x_1 \sim p_{\text{data}},\qquad
t \sim \text{Uniform}(0,1)
$$

这条路径的目标速度为：

$$
u_t = \frac{dx_t}{dt} = x_1 - x_0
$$

模型训练目标为：

$$
\mathcal{L}_{\text{FM}}
=
\mathbb{E}_{x_0,x_1,t}
\left[
\left\|
v_\theta(x_t,t) - (x_1-x_0)
\right\|_2^2
\right]
$$

采样时从 \(x(0)\sim\mathcal{N}(0,I)\) 开始，用 Euler 方法求解 ODE：

$$
x_{k+1} = x_k + v_\theta(x_k,t_k)\Delta t
$$

其中：

$$
t_k = \frac{k}{N},\qquad
\Delta t = \frac{1}{N}
$$

## Diffusion 与 Flow Matching 对比

| 项目 | Diffusion / DDPM | Flow Matching / Rectified Flow |
| ---- | ---------------- | ------------------------------ |
| 核心思想 | 学会逐步去噪 | 学习从噪声到数据的速度场 |
| 训练数据构造 | 对真实图像按随机时间步加噪 | 随机配对噪声图像 \(x_0\) 和真实图像 \(x_1\)，构造插值路径 |
| 模型预测目标 | 噪声 \(\epsilon_\theta(x_t,t)\) | 速度 \(v_\theta(x_t,t)\) |
| 时间变量 | 离散时间步 \(t \in \{0,\dots,T-1\}\) | 连续时间 \(t \in [0,1]\) |
| 采样过程 | 从 \(x_T\) 到 \(x_0\) 反向去噪 | 从 \(t=0\) 到 \(t=1\) 做 ODE 积分 |
| 数学形式 | 随机反向马尔可夫链 | 确定性或半确定性连续流 |
| 直观理解 | 从噪声里一步步洗出图像 | 让噪声沿速度场流动到图像 |
| 优点 | 理论成熟，效果稳定 | 形式简洁，采样步数可较少 |
| 局限 | 采样通常需要较多步 | 简单线性路径在复杂数据上可能不是最优 |

## 项目结构说明

```text
image_flow_diffusion_demo/
├── README.md
├── requirements.txt
├── configs.py
├── data.py
├── models.py
├── train_diffusion.py
├── train_flow_matching.py
├── sample_diffusion.py
├── sample_flow_matching.py
├── visualize.py
├── utils.py
└── outputs/
    ├── checkpoints/
    ├── samples/
    └── logs/
```

- `configs.py`：默认参数配置。
- `data.py`：MNIST、FashionMNIST、synthetic_shapes 数据加载。
- `models.py`：带 sinusoidal time embedding 的轻量级 `TinyUNet`。
- `train_diffusion.py`：DDPM 训练脚本。
- `sample_diffusion.py`：DDPM 反向采样脚本。
- `train_flow_matching.py`：Flow Matching 训练脚本。
- `sample_flow_matching.py`：Flow Matching ODE 采样脚本。
- `visualize.py`：图像网格、过程图、loss 曲线保存工具。
- `utils.py`：随机种子、设备选择、输出目录、checkpoint 和 beta schedule 工具。

## 运行命令

进入项目目录：

```bash
cd image_flow_diffusion_demo
```

训练 Diffusion：

```bash
python train_diffusion.py --dataset mnist --epochs 5
```

采样 Diffusion：

```bash
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt
```

训练 Flow Matching：

```bash
python train_flow_matching.py --dataset mnist --epochs 5
```

采样 Flow Matching：

```bash
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt
```

如果当前环境不能下载 MNIST，可以使用离线合成数据集：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 5
python train_flow_matching.py --dataset synthetic_shapes --epochs 5
```

验收用快速流程：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 2
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt

python train_flow_matching.py --dataset synthetic_shapes --epochs 2
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt
```

如果只是想快速检查脚本是否跑通，可以减少 batch 数：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 1 --max_batches 2 --num_workers 0
python train_flow_matching.py --dataset synthetic_shapes --epochs 1 --max_batches 2 --num_workers 0
```

## 结果说明

训练和采样会自动创建 `outputs/` 目录：

- `outputs/checkpoints/`：保存模型权重。
- `outputs/samples/`：保存生成图像和中间过程图。
- `outputs/logs/`：保存 loss 曲线。

完整流程后应看到：

- `outputs/checkpoints/diffusion.pt`
- `outputs/checkpoints/flow_matching.pt`
- `outputs/logs/diffusion_loss.png`
- `outputs/logs/flow_matching_loss.png`
- `outputs/samples/diffusion_samples.png`
- `outputs/samples/flow_matching_samples.png`
- `outputs/samples/diffusion_denoising_process.png`
- `outputs/samples/flow_matching_generation_process.png`

> **提示**：训练过程中还会生成 `diffusion_epoch_*.png` 和 `flow_matching_epoch_*.png` 等中间采样图。这些文件只是用于观察训练进度，最终保留上面 8 个文件即可，中间文件可以删除以节省空间。

> **Windows 用户注意**：在部分 Windows 环境下，`matplotlib` 保存中文标题时可能会报 `Glyph ... missing from font(s)` 的警告。这不会影响训练、采样和保存的图片，只是图中文字显示可能不完整。如需完美显示中文，可以安装支持中文的 matplotlib 字体并配置 `matplotlibrc`。

## 小白理解版

**Diffusion**：把真实图像逐渐加噪，训练模型学会把噪声去掉。生成时从纯噪声开始，一步步去噪，最后得到图像。

**Flow Matching**：从噪声图像和真实图像之间连一条路径，训练模型学习每个中间状态应该往哪里变化。生成时从噪声开始，沿着模型预测的速度场流动到图像。

## 5090 大任务配置

当前版本已经把默认任务升级到更适合高端 GPU 的设置：

- `image_size=64`
- `dataset_size=50000`
- `model_channels=64`
- `num_downs=3`
- `use_attention=True`
- `diffusion num_timesteps=500`
- `flow sample_steps=100`
- 默认开启 CUDA AMP 混合精度

推荐先跑 synthetic_shapes 大一点的离线任务：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --num_timesteps 500 --sample_every 2
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt --num_samples 64

python train_flow_matching.py --dataset synthetic_shapes --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --sample_steps 100 --sample_every 2
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt --num_samples 64 --sample_steps 100
```

如果联网方便，也可以用 `fashion_mnist` 或 `mnist` 跑真实数据集，视觉效果通常比 synthetic_shapes 更丰富（以 fashion_mnist 为例）：

```bash
python train_diffusion.py --dataset fashion_mnist --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --num_timesteps 500 --sample_every 2
python sample_diffusion.py --checkpoint outputs/checkpoints/diffusion.pt --num_samples 64

python train_flow_matching.py --dataset fashion_mnist --epochs 20 --batch_size 128 --dataset_size 50000 --image_size 64 --model_channels 64 --num_downs 3 --sample_steps 100 --sample_every 2
python sample_flow_matching.py --checkpoint outputs/checkpoints/flow_matching.pt --num_samples 64 --sample_steps 100
```

如果显存还有余量，可以继续加大：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 50 --batch_size 256 --dataset_size 100000 --image_size 64 --model_channels 96 --num_downs 3 --num_timesteps 1000 --sample_every 5
python train_flow_matching.py --dataset synthetic_shapes --epochs 50 --batch_size 256 --dataset_size 100000 --image_size 64 --model_channels 96 --num_downs 3 --sample_steps 150 --sample_every 5
```

如果想退回原来的轻量 demo，可以这样跑：

```bash
python train_diffusion.py --dataset synthetic_shapes --epochs 5 --batch_size 128 --dataset_size 10000 --image_size 28 --model_channels 32 --num_downs 1 --no-use_attention --num_timesteps 200
python train_flow_matching.py --dataset synthetic_shapes --epochs 5 --batch_size 128 --dataset_size 10000 --image_size 28 --model_channels 32 --num_downs 1 --no-use_attention --sample_steps 50
```
