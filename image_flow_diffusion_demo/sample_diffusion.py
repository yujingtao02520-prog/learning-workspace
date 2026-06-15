import argparse
import os

import torch
from tqdm import tqdm

from configs import DEFAULT_CONFIG
from models import TinyUNet
from utils import build_linear_beta_schedule, ensure_output_dirs, exists_or_raise, extract, get_device
from visualize import save_image_grid, save_process_grid


def sample_ddpm(
    model: TinyUNet,
    num_samples: int,
    image_size: int,
    num_timesteps: int,
    device: torch.device,
    betas: torch.Tensor = None,
    alphas: torch.Tensor = None,
    alpha_bars: torch.Tensor = None,
    process_samples: int = 8,
):
    """
    DDPM 反向采样。

    从 x_T ~ N(0, I) 开始，逐步用模型预测噪声并按闭式更新得到 x_{t-1}。
    """

    model.eval()
    if betas is None or alphas is None or alpha_bars is None:
        betas, alphas, alpha_bars = build_linear_beta_schedule(num_timesteps)
        betas = betas.to(device)
        alphas = alphas.to(device)
        alpha_bars = alpha_bars.to(device)

    x = torch.randn(num_samples, 1, image_size, image_size, device=device)
    process = [x[:process_samples].detach().cpu()]
    save_points = set(torch.linspace(num_timesteps - 1, 0, steps=8).long().tolist())

    for step in tqdm(reversed(range(num_timesteps)), total=num_timesteps, desc="DDPM Sampling"):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        t_norm = t.float() / max(num_timesteps - 1, 1)

        # epsilon_theta(x_t, t)：模型估计当前图像里包含的噪声。
        pred_noise = model(x, t_norm)

        beta_t = extract(betas, t, x.shape)
        alpha_t = extract(alphas, t, x.shape)
        alpha_bar_t = extract(alpha_bars, t, x.shape)

        # DDPM 反向均值：
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - beta_t / torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-8) * pred_noise
        )

        if step > 0:
            # t > 0 时加入高斯噪声，近似从 p_theta(x_{t-1}|x_t) 采样。
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            # 最后一步不再加噪声。
            x = mean

        if step in save_points:
            process.append(x[:process_samples].detach().cpu())

    return x.detach().cpu().clamp(-1, 1), process


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 DDPM checkpoint 采样")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/diffusion.pt")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--model_channels", type=int, default=None)
    parser.add_argument("--num_downs", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--use_attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ensure_output_dirs(args.output_dir)
    exists_or_raise(args.checkpoint)
    torch.manual_seed(args.seed)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", DEFAULT_CONFIG.image_size))
    model_channels = args.model_channels or int(config.get("model_channels", DEFAULT_CONFIG.model_channels))
    num_downs = args.num_downs or int(config.get("num_downs", DEFAULT_CONFIG.num_downs))
    use_attention = (
        args.use_attention if args.use_attention is not None else bool(config.get("use_attention", DEFAULT_CONFIG.use_attention))
    )
    dropout = args.dropout if args.dropout is not None else float(config.get("dropout", 0.0))
    num_timesteps = args.num_timesteps or int(
        config.get("num_timesteps", DEFAULT_CONFIG.diffusion_num_timesteps)
    )

    model = TinyUNet(
        in_channels=1,
        out_channels=1,
        model_channels=model_channels,
        num_downs=num_downs,
        use_attention=use_attention,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        samples, process = sample_ddpm(
            model=model,
            num_samples=args.num_samples,
            image_size=image_size,
            num_timesteps=num_timesteps,
            device=device,
            process_samples=min(8, args.num_samples),
        )

    save_image_grid(samples, os.path.join(paths["samples"], "diffusion_samples.png"), nrow=8)
    save_process_grid(process, os.path.join(paths["samples"], "diffusion_denoising_process.png"))
    print(f"采样完成，结果保存在 {paths['samples']}")


if __name__ == "__main__":
    main()
