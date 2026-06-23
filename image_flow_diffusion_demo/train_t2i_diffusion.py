import argparse
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_t2i import get_t2i_dataloader, VOCAB, MNIST_PROMPT_MAP, FASHION_MNIST_PROMPT_MAP
from models_t2i import DiT
from sample_t2i_diffusion import sample_t2i_ddpm
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
    parser = argparse.ArgumentParser(description="训练文生图 Diffusion (DDPM) 模型")
    parser.add_argument("--dataset", type=str, default="synthetic_shapes", choices=["synthetic_shapes", "mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_timesteps", type=int, default=400)
    parser.add_argument("--image_size", type=int, default=32, help="训练图像分辨率（建议 32 或 64）")
    parser.add_argument("--patch_size", type=int, default=4, help="Transformer Patch 大小")
    parser.add_argument("--hidden_size", type=int, default=128, help="Transformer 特征维度")
    parser.add_argument("--num_layers", type=int, default=4, help="DiT 块的层数")
    parser.add_argument("--num_heads", type=int, default=4, help="多头注意力的头数")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dataset_size", type=int, default=30000)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--cfg_prob", type=float, default=0.1, help="训练中随机抛弃文本条件（做无条件训练）的概率")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)
    paths = ensure_output_dirs(args.output_dir)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"

    dataloader = get_t2i_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        dataset_size=args.dataset_size,
    )

    # 初始化文生图 DiT 模型
    model = DiT(
        in_channels=1,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=len(VOCAB),
        image_size=args.image_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    betas, alphas, alpha_bars = build_linear_beta_schedule(args.num_timesteps)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    losses = []
    global_step = 0

    # 确定中途可视化的 Prompt 列表
    if args.dataset == "synthetic_shapes":
        sample_prompts = ["a white circle", "a white square", "a white triangle", "a white diamond", "a white line", "a white plus"]
    elif args.dataset in {"fashion_mnist", "fashionmnist"}:
        sample_prompts = [FASHION_MNIST_PROMPT_MAP[i] for i in range(8)]
    else:
        sample_prompts = [MNIST_PROMPT_MAP[i] for i in range(8)]

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(dataloader, desc=f"T2I Diffusion Epoch {epoch}/{args.epochs}")

        for batch_idx, (x0, cond_tokens, _) in enumerate(progress):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            x0 = x0.to(device, non_blocking=True)
            cond_tokens = cond_tokens.to(device, non_blocking=True)
            batch_size = x0.shape[0]

            # 随机进行无条件训练（Classifier-Free Guidance 核心）
            if args.cfg_prob > 0:
                mask = torch.rand(batch_size, device=device) < args.cfg_prob
                # 将文本 Token 替换成 0（对应 <pad>）
                cond_tokens[mask] = 0

            # 离散加噪步
            t = torch.randint(0, args.num_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)

            # 加噪
            sqrt_alpha_bar = torch.sqrt(extract(alpha_bars, t, x0.shape))
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - extract(alpha_bars, t, x0.shape))
            xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

            t_norm = t.float() / max(args.num_timesteps - 1, 1)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 输入图像 xt、时间 t_norm 和 文本标记 cond_tokens
                pred_noise = model(xt, t_norm, cond_tokens)
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
                plot_loss(losses, os.path.join(paths["logs"], "t2i_diffusion_loss.png"), "T2I DDPM Training Loss")

        should_sample = args.sample_every > 0 and epoch % args.sample_every == 0
        should_save = args.save_every > 0 and epoch % args.save_every == 0

        if should_sample:
            model.eval()
            print(f"\nSampling epoch {epoch}...")
            with torch.no_grad():
                samples, process = sample_t2i_ddpm(
                    model=model,
                    prompts=sample_prompts,
                    image_size=args.image_size,
                    num_timesteps=args.num_timesteps,
                    device=device,
                    betas=betas,
                    alphas=alphas,
                    alpha_bars=alpha_bars,
                    cfg_scale=3.0,
                    num_samples_per_prompt=4,
                    process_samples=8
                )
            # 保存多类别 Prompt 结果网格
            save_image_grid(samples, os.path.join(paths["samples"], f"t2i_diffusion_epoch_{epoch}.png"), nrow=4)
            save_process_grid(process, os.path.join(paths["samples"], "t2i_diffusion_denoising_process.png"))

        plot_loss(losses, os.path.join(paths["logs"], "t2i_diffusion_loss.png"), "T2I DDPM Training Loss")

        if should_save or epoch == args.epochs:
            save_checkpoint(
                os.path.join(paths["checkpoints"], "t2i_diffusion.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=vars(args),
            )

    print(f"训练完成！模型权重保存在 {os.path.join(paths['checkpoints'], 't2i_diffusion.pt')}")

if __name__ == "__main__":
    train()
