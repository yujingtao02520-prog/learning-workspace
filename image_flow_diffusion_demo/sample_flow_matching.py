import argparse
import os

import torch
from tqdm import tqdm

from configs import DEFAULT_CONFIG
from models import TinyUNet
from utils import ensure_output_dirs, exists_or_raise, get_device
from visualize import save_image_grid, save_process_grid


def sample_flow_matching(
    model: TinyUNet,
    num_samples: int,
    image_size: int,
    sample_steps: int,
    device: torch.device,
    process_samples: int = 8,
    method: str = "euler",
):
    """
    Flow Matching ODE 采样。

    从 x(0) ~ N(0, I) 开始，沿 dx/dt = v_theta(x, t) 积分到 t=1。
    """

    model.eval()
    x = torch.randn(num_samples, 1, image_size, image_size, device=device)
    dt = 1.0 / sample_steps
    process = [x[:process_samples].detach().cpu()]
    save_points = set(torch.linspace(0, sample_steps, steps=8).long().tolist())

    for step in tqdm(range(sample_steps), desc="Flow Matching Sampling"):
        t = torch.full((num_samples,), step / sample_steps, device=device)

        # Euler 更新：
        # x_{k+1} = x_k + v_theta(x_k, t_k) * Delta t
        velocity = model(x, t)
        if method == "heun":
            x_euler = x + velocity * dt
            t_next = torch.full((num_samples,), min((step + 1) / sample_steps, 1.0), device=device)
            velocity_next = model(x_euler, t_next)
            x = x + 0.5 * (velocity + velocity_next) * dt
        else:
            x = x + velocity * dt

        if (step + 1) in save_points:
            process.append(x[:process_samples].detach().cpu())

    return x.detach().cpu().clamp(-1, 1), process


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 Flow Matching checkpoint 采样")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/flow_matching.pt")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--model_channels", type=int, default=None)
    parser.add_argument("--num_downs", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--use_attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
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
    sample_steps = args.sample_steps or int(config.get("sample_steps", DEFAULT_CONFIG.flow_sample_steps))

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
        samples, process = sample_flow_matching(
            model=model,
            num_samples=args.num_samples,
            image_size=image_size,
            sample_steps=sample_steps,
            device=device,
            process_samples=min(8, args.num_samples),
            method=args.method,
        )

    save_image_grid(samples, os.path.join(paths["samples"], "flow_matching_samples.png"), nrow=8)
    save_process_grid(process, os.path.join(paths["samples"], "flow_matching_generation_process.png"))
    print(f"采样完成，结果保存在 {paths['samples']}")


if __name__ == "__main__":
    main()
