import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_images(images: torch.Tensor) -> np.ndarray:
    """把 [-1, 1] 的图像张量转成 [0, 1] 的 numpy 数组。"""

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    images = (images.clamp(-1, 1) + 1.0) / 2.0
    return images.numpy()


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """保存图像网格，输入 shape 为 [B, 1, H, W]。"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    images_np = _to_numpy_images(images)
    num_images = images_np.shape[0]
    ncol = int(np.ceil(num_images / nrow))

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.4, ncol * 1.4))
    axes = np.array(axes).reshape(ncol, nrow)
    for idx in range(ncol * nrow):
        ax = axes[idx // nrow, idx % nrow]
        ax.axis("off")
        if idx < num_images:
            ax.imshow(images_np[idx, 0], cmap="gray", vmin=0, vmax=1)
    plt.tight_layout(pad=0.05)
    plt.savefig(path, dpi=160)
    plt.close(fig)


def plot_loss(losses: List[float], path: str, title: str) -> None:
    """保存训练 loss 曲线。"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("记录步数")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_process_grid(process_images: List[torch.Tensor], path: str) -> None:
    """
    保存采样过程图。

    process_images 是若干时间点的图像列表，每个元素 shape 为 [B, 1, H, W]。
    横向展示时间变化，纵向展示不同样本。
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(process_images) == 0:
        raise ValueError("process_images 不能为空。")

    stacked = torch.stack([img.detach().cpu() for img in process_images], dim=1)
    stacked_np = _to_numpy_images(stacked.reshape(-1, *stacked.shape[2:]))
    batch_size = stacked.shape[0]
    num_steps = stacked.shape[1]

    fig, axes = plt.subplots(batch_size, num_steps, figsize=(num_steps * 1.3, batch_size * 1.3))
    axes = np.array(axes).reshape(batch_size, num_steps)
    for row in range(batch_size):
        for col in range(num_steps):
            ax = axes[row, col]
            ax.axis("off")
            ax.imshow(stacked_np[row * num_steps + col, 0], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(f"{col + 1}", fontsize=8)
    plt.tight_layout(pad=0.05)
    plt.savefig(path, dpi=160)
    plt.close(fig)
