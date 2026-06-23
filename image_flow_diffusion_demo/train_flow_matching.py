import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import DEFAULT_CONFIG
from data import get_dataloader
from models import TinyUNet
from sample_flow_matching import sample_flow_matching
from utils import ensure_output_dirs, get_device, save_checkpoint, set_seed
from visualize import plot_loss, save_image_grid, save_process_grid


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Flow Matching / Rectified Flow 模型")
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG.dataset)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--sample_steps", type=int, default=DEFAULT_CONFIG.flow_sample_steps)
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
    parser.add_argument("--sample_method", type=str, default="euler", choices=["euler", "heun"])
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

    losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(dataloader, desc=f"Flow Matching Epoch {epoch}/{args.epochs}")

        for batch_idx, (x1, _) in enumerate(progress):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            x1 = x1.to(device, non_blocking=True)
            batch_size = x1.shape[0]
            x0 = torch.randn_like(x1)

            # t 是连续时间，范围 [0, 1]。
            t = torch.rand(batch_size, device=device)
            t_img = t[:, None, None, None]

            # 线性插值路径：x_t = (1 - t) * x_0 + t * x_1
            xt = (1.0 - t_img) * x0 + t_img * x1

            # 目标速度是路径对时间的导数：u_t = d x_t / dt = x_1 - x_0
            target_velocity = x1 - x0
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_velocity = model(xt, t)
                # Flow Matching 损失：L = || v_theta(x_t, t) - (x_1 - x_0) ||_2^2
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
                plot_loss(
                    losses,
                    os.path.join(paths["logs"], "flow_matching_loss.png"),
                    "Flow Matching Training Loss",
                )

        should_sample = args.sample_every > 0 and epoch % args.sample_every == 0
        should_save = args.save_every > 0 and epoch % args.save_every == 0

        if should_sample:
            model.eval()
            with torch.no_grad():
                samples, process = sample_flow_matching(
                    model=model,
                    num_samples=args.sample_num,
                    image_size=args.image_size,
                    sample_steps=args.sample_steps,
                    device=device,
                    process_samples=min(8, args.sample_num),
                    method=args.sample_method,
                )
            save_image_grid(samples, os.path.join(paths["samples"], f"flow_matching_epoch_{epoch}.png"), nrow=8)
            save_process_grid(process, os.path.join(paths["samples"], "flow_matching_generation_process.png"))

        plot_loss(losses, os.path.join(paths["logs"], "flow_matching_loss.png"), "Flow Matching Training Loss")

        if should_save or epoch == args.epochs:
            save_checkpoint(
                os.path.join(paths["checkpoints"], "flow_matching.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=vars(args),
            )

    print(f"训练完成，checkpoint 已保存到 {os.path.join(paths['checkpoints'], 'flow_matching.pt')}")


if __name__ == "__main__":
    train()
