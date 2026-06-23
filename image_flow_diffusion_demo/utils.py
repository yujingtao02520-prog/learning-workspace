import os
import random
from typing import Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，方便复现实验结果。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自动选择 CUDA 或 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """确保目录存在。"""

    os.makedirs(path, exist_ok=True)


def ensure_output_dirs(output_dir: str = "./outputs") -> Dict[str, str]:
    """创建训练和采样需要的输出目录。"""

    paths = {
        "root": output_dir,
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "samples": os.path.join(output_dir, "samples"),
        "logs": os.path.join(output_dir, "logs"),
    }
    for path in paths.values():
        ensure_dir(path)
    return paths


def exists_or_raise(path: str) -> None:
    """采样脚本加载 checkpoint 前给出清晰错误。"""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {path}\n"
            "请先运行对应训练脚本，例如 train_diffusion.py 或 train_flow_matching.py。"
        )


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: Iterable[int]) -> torch.Tensor:
    """按 batch 中的时间步取出系数，并 reshape 成可广播到图像张量的形状。"""

    batch_size = timesteps.shape[0]
    values = values.to(timesteps.device)
    out = values.gather(0, timesteps)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def save_checkpoint(path: str, model, optimizer, epoch: int, config: Dict) -> None:
    """保存包含模型、优化器、epoch 和配置的 checkpoint。"""

    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def build_linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """DDPM 使用的线性 beta schedule。"""

    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
