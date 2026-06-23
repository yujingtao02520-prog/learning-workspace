import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import DEFAULT_CONFIG
from data import get_dataloader
from models import TinyUNet
from sample_diffusion import sample_ddpm
from utils import (
    build_linear_beta_schedule,
    ensure_output_dirs,
    extract,
    get_device,
    save_checkpoint,
    set_seed,
)
from visualize import plot_loss, save_image_grid, save_process_grid


def parse_args():
    parser = argparse.ArgumentParser(description="训练 DDPM 图像生成模型")
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG.dataset)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--num_timesteps", type=int, default=DEFAULT_CONFIG.diffusion_num_timesteps)
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG.image_size)
    parser.add_argument("--model_channels", type=int, default=DEFAULT_CONFIG.model_channels)
    parser.add_argument("--num_downs", type=int, default=DEFAULT_CONFIG.num_downs, choices=[1, 2, 3])
    parser.add_argument("--use_attention", action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG.use_attention)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=DEFAULT_CONFIG.dataset_size)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=DEFAULT_CONFIG.sample_every)
    parser.add_argument("--save_every", type=int, default=DEFAULT_CONFIG.save_every)
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG.amp)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_CONFIG.grad_clip)
    parser.add_argument("--max_batches", type=int, default=0, help="调试用：每个 epoch 最多训练多少 batch，0 表示不限制")
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)
    paths = ensure_output_dirs(args.output_dir)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"

    dataloader = get_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        dataset_size=args.dataset_size,
    )

    model = TinyUNet(
        in_channels=1,
        out_channels=1,
        model_channels=args.model_channels,
        num_downs=args.num_downs,
        use_attention=args.use_attention,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    betas, alphas, alpha_bars = build_linear_beta_schedule(args.num_timesteps)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(dataloader, desc=f"DDPM Epoch {epoch}/{args.epochs}")

        for batch_idx, (x0, _) in enumerate(progress):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            x0 = x0.to(device, non_blocking=True)
            batch_size = x0.shape[0]

            # t 是离散扩散步，范围为 [0, T-1]。
            t = torch.randint(0, args.num_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)

            # 前向加噪公式：
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            sqrt_alpha_bar = torch.sqrt(extract(alpha_bars, t, x0.shape))
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - extract(alpha_bars, t, x0.shape))
            xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

            t_norm = t.float() / max(args.num_timesteps - 1, 1)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 模型预测噪声 epsilon_theta(x_t, t)。
                pred_noise = model(xt, t_norm)
                # DDPM 损失：L = || epsilon_theta(x_t, t) - epsilon ||_2^2
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.log_interval == 0:
                plot_loss(losses, os.path.join(paths["logs"], "diffusion_loss.png"), "DDPM Training Loss")

        should_sample = args.sample_every > 0 and epoch % args.sample_every == 0
        should_save = args.save_every > 0 and epoch % args.save_every == 0

        if should_sample:
            model.eval()
            with torch.no_grad():
                samples, process = sample_ddpm(
                    model=model,
                    num_samples=args.sample_num,
                    image_size=args.image_size,
                    num_timesteps=args.num_timesteps,
                    device=device,
                    betas=betas,
                    alphas=alphas,
                    alpha_bars=alpha_bars,
                    process_samples=min(8, args.sample_num),
                )
            save_image_grid(samples, os.path.join(paths["samples"], f"diffusion_epoch_{epoch}.png"), nrow=8)
            save_process_grid(process, os.path.join(paths["samples"], "diffusion_denoising_process.png"))

        plot_loss(losses, os.path.join(paths["logs"], "diffusion_loss.png"), "DDPM Training Loss")

        if should_save or epoch == args.epochs:
            save_checkpoint(
                os.path.join(paths["checkpoints"], "diffusion.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=vars(args),
            )

    print(f"训练完成，checkpoint 已保存到 {os.path.join(paths['checkpoints'], 'diffusion.pt')}")


if __name__ == "__main__":
    train()
