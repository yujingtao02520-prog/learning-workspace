import argparse
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_t2i import get_t2i_dataloader, VOCAB, MNIST_PROMPT_MAP, FASHION_MNIST_PROMPT_MAP
from models_t2i import DiT
from sample_t2i_flow import sample_t2i_flow
from utils import (
    ensure_output_dirs,
    get_device,
    save_checkpoint,
    set_seed,
)
from visualize import plot_loss, save_image_grid, save_process_grid

def parse_args():
    parser = argparse.ArgumentParser(description="训练文生图 Flow Matching / Rectified Flow 模型")
    parser.add_argument("--dataset", type=str, default="synthetic_shapes", choices=["synthetic_shapes", "mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample_steps", type=int, default=80)
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
    parser.add_argument("--cfg_prob", type=float, default=0.1, help="训练中随机抛弃文本条件的概率")
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
        progress = tqdm(dataloader, desc=f"T2I Flow Matching Epoch {epoch}/{args.epochs}")

        for batch_idx, (x1, cond_tokens, _) in enumerate(progress):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            x1 = x1.to(device, non_blocking=True)
            cond_tokens = cond_tokens.to(device, non_blocking=True)
            batch_size = x1.shape[0]

            # 随机进行无条件训练（Classifier-Free Guidance 核心）
            if args.cfg_prob > 0:
                mask = torch.rand(batch_size, device=device) < args.cfg_prob
                # 将文本 Token 替换成 0（对应 <pad>）
                cond_tokens[mask] = 0

            # 采样随机高斯噪声 x0 作为起点
            x0 = torch.randn_like(x1)

            # 随机采样时间步 t ∈ [0, 1]
            t = torch.rand(batch_size, device=device)
            t_img = t[:, None, None, None]

            # 线性路径插值：x_t = (1 - t) * x_0 + t * x_1
            xt = (1.0 - t_img) * x0 + t_img * x1

            # 目标速度场：u_t = dx_t/dt = x_1 - x_0
            target_velocity = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 输入图像 xt、时间步 t 和文本特征 cond_tokens，预测当前状态的速度场
                pred_velocity = model(xt, t, cond_tokens)
                loss = F.mse_loss(pred_velocity, target_velocity)

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
                plot_loss(losses, os.path.join(paths["logs"], "t2i_flow_matching_loss.png"), "T2I Flow Matching Training Loss")

        should_sample = args.sample_every > 0 and epoch % args.sample_every == 0
        should_save = args.save_every > 0 and epoch % args.save_every == 0

        if should_sample:
            model.eval()
            print(f"\nSampling epoch {epoch}...")
            with torch.no_grad():
                samples, process = sample_t2i_flow(
                    model=model,
                    prompts=sample_prompts,
                    image_size=args.image_size,
                    sample_steps=args.sample_steps,
                    device=device,
                    cfg_scale=3.0,
                    num_samples_per_prompt=4,
                    process_samples=8,
                    method="euler"
                )
            # 保存多类别 Prompt 结果网格
            save_image_grid(samples, os.path.join(paths["samples"], f"t2i_flow_epoch_{epoch}.png"), nrow=4)
            save_process_grid(process, os.path.join(paths["samples"], "t2i_flow_generation_process.png"))

        plot_loss(losses, os.path.join(paths["logs"], "t2i_flow_matching_loss.png"), "T2I Flow Matching Training Loss")

        if should_save or epoch == args.epochs:
            save_checkpoint(
                os.path.join(paths["checkpoints"], "t2i_flow.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=vars(args),
            )

    print(f"训练完成！模型权重保存在 {os.path.join(paths['checkpoints'], 't2i_flow.pt')}")

if __name__ == "__main__":
    train()
