import argparse
import os
import torch
from tqdm import tqdm

from data_t2i import tokenize_prompt, VOCAB
from models_t2i import DiT
from utils import ensure_output_dirs, exists_or_raise, get_device
from visualize import save_image_grid, save_process_grid

def sample_t2i_flow(
    model: DiT,
    prompts: list,
    image_size: int,
    sample_steps: int,
    device: torch.device,
    cfg_scale: float = 3.0,
    num_samples_per_prompt: int = 4,
    process_samples: int = 8,
    method: str = "euler"
):
    """
    文生图 Flow Matching ODE 积分采样（含 Classifier-Free Guidance）。
    从 x(0) ~ N(0, I) 逐步积分到 x(1) 得到图片。
    每一行对应一个 prompt 的生成。
    """
    model.eval()
    num_prompts = len(prompts)
    total_samples = num_prompts * num_samples_per_prompt
    
    # 扩展 prompts
    flat_prompts = []
    for p in prompts:
        flat_prompts.extend([p] * num_samples_per_prompt)
        
    cond_tokens = torch.stack([tokenize_prompt(p) for p in flat_prompts], dim=0).to(device)
    uncond_tokens = torch.zeros_like(cond_tokens).to(device)  # 对应无条件的 <pad> token
    
    # 从纯噪声开始
    x = torch.randn(total_samples, 1, image_size, image_size, device=device)
    dt = 1.0 / sample_steps
    process = [x[:process_samples].detach().cpu()]
    save_points = set(torch.linspace(0, sample_steps, steps=8).long().tolist())

    for step in tqdm(range(sample_steps), desc="T2I Flow Matching Sampling"):
        t = torch.full((total_samples,), step / sample_steps, device=device)

        # 运用 CFG：拼接条件与无条件样本
        if cfg_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            text_in = torch.cat([cond_tokens, uncond_tokens], dim=0)
            
            pred = model(x_in, t_in, text_in)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            
            velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            velocity = model(x, t, cond_tokens)

        if method == "heun":
            # 预估器（Euler 预估）
            x_euler = x + velocity * dt
            t_next = torch.full((total_samples,), min((step + 1) / sample_steps, 1.0), device=device)
            
            # 在预估位置预测速度
            if cfg_scale > 1.0:
                x_in = torch.cat([x_euler, x_euler], dim=0)
                t_in = torch.cat([t_next, t_next], dim=0)
                text_in = torch.cat([cond_tokens, uncond_tokens], dim=0)
                
                pred = model(x_in, t_in, text_in)
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                velocity_next = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                velocity_next = model(x_euler, t_next, cond_tokens)
                
            # 校正器（Heun 校正）
            x = x + 0.5 * (velocity + velocity_next) * dt
        else:
            # 纯 Euler 方法
            x = x + velocity * dt

        if (step + 1) in save_points:
            process.append(x[:process_samples].detach().cpu())

    return x.detach().cpu().clamp(-1, 1), process

def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 T2I DiT Flow Matching 模型采样")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/t2i_flow.pt")
    parser.add_argument("--prompt", type=str, default="a white circle", help="需要生成的文本描述")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance 引导强度")
    parser.add_argument("--sample_steps", type=int, default=80, help="ODE 积分步数")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    paths = ensure_output_dirs(args.output_dir)
    exists_or_raise(args.checkpoint)
    torch.manual_seed(args.seed)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    
    image_size = args.image_size or int(config.get("image_size", 64))
    patch_size = args.patch_size or int(config.get("patch_size", 4))
    hidden_size = args.hidden_size or int(config.get("hidden_size", 128))
    num_layers = args.num_layers or int(config.get("num_layers", 4))
    num_heads = args.num_heads or int(config.get("num_heads", 4))
    sample_steps = args.sample_steps or int(config.get("sample_steps", 80))

    model = DiT(
        in_channels=1,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=len(VOCAB),
        image_size=image_size
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Generating images for prompt: '{args.prompt}' using {args.method} method")

    with torch.no_grad():
        samples, process = sample_t2i_flow(
            model=model,
            prompts=[args.prompt],
            image_size=image_size,
            sample_steps=sample_steps,
            device=device,
            cfg_scale=args.cfg_scale,
            num_samples_per_prompt=args.num_samples,
            process_samples=min(8, args.num_samples),
            method=args.method
        )

    # 保存图片网格
    save_image_grid(samples, os.path.join(paths["samples"], "t2i_flow_samples.png"), nrow=4)
    save_process_grid(process, os.path.join(paths["samples"], "t2i_flow_generation_process.png"))
    print(f"采样成功！结果已保存到 {paths['samples']}")

if __name__ == "__main__":
    main()
