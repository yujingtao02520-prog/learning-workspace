import argparse
import os
import torch
from tqdm import tqdm

from data_t2i import tokenize_prompt, VOCAB, MNIST_PROMPT_MAP, FASHION_MNIST_PROMPT_MAP
from models_t2i import DiT
from utils import build_linear_beta_schedule, ensure_output_dirs, exists_or_raise, extract, get_device
from visualize import save_image_grid, save_process_grid

def sample_t2i_ddpm(
    model: DiT,
    prompts: list,
    image_size: int,
    num_timesteps: int,
    device: torch.device,
    betas: torch.Tensor = None,
    alphas: torch.Tensor = None,
    alpha_bars: torch.Tensor = None,
    cfg_scale: float = 3.0,
    num_samples_per_prompt: int = 4,
    process_samples: int = 8
):
    """
    文生图 DDPM 反向采样（含 Classifier-Free Guidance）。
    每一行对应一个 prompt 的生成。
    """
    model.eval()
    if betas is None or alphas is None or alpha_bars is None:
        betas, alphas, alpha_bars = build_linear_beta_schedule(num_timesteps)
        betas = betas.to(device)
        alphas = alphas.to(device)
        alpha_bars = alpha_bars.to(device)

    num_prompts = len(prompts)
    total_samples = num_prompts * num_samples_per_prompt
    
    # 扩展 prompts 到对应的 sample 数
    flat_prompts = []
    for p in prompts:
        flat_prompts.extend([p] * num_samples_per_prompt)
        
    # 分词
    cond_tokens = torch.stack([tokenize_prompt(p) for p in flat_prompts], dim=0).to(device)
    uncond_tokens = torch.zeros_like(cond_tokens).to(device)  # 用 0 (即 <pad>) 表示无条件提示词
    
    # 从纯噪声开始
    x = torch.randn(total_samples, 1, image_size, image_size, device=device)
    
    # 记录过程：只记录前 process_samples 个样本的去噪路径
    process = [x[:process_samples].detach().cpu()]
    save_points = set(torch.linspace(num_timesteps - 1, 0, steps=8).long().tolist())

    for step in tqdm(reversed(range(num_timesteps)), total=num_timesteps, desc="T2I DDPM Sampling"):
        t = torch.full((total_samples,), step, device=device, dtype=torch.long)
        t_norm = t.float() / max(num_timesteps - 1, 1)

        # 运用 CFG：拼接条件与无条件样本做一次 Batch Forward 提高速度
        if cfg_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_norm, t_norm], dim=0)
            text_in = torch.cat([cond_tokens, uncond_tokens], dim=0)
            
            pred = model(x_in, t_in, text_in)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            
            # 外推：pred = pred_uncond + s * (pred_cond - pred_uncond)
            pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = model(x, t_norm, cond_tokens)

        beta_t = extract(betas, t, x.shape)
        alpha_t = extract(alphas, t, x.shape)
        alpha_bar_t = extract(alpha_bars, t, x.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - beta_t / torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-8) * pred_noise
        )

        if step > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean

        if step in save_points:
            process.append(x[:process_samples].detach().cpu())

    return x.detach().cpu().clamp(-1, 1), process

def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 T2I DiT 扩散模型采样")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/t2i_diffusion.pt")
    parser.add_argument("--prompt", type=str, default="a white circle", help="需要生成的文本描述")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance 引导强度")
    parser.add_argument("--num_samples", type=int, default=16, help="总生成数量")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
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
    num_timesteps = int(config.get("num_timesteps", 500))

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
    print(f"Generating images for prompt: '{args.prompt}'")

    with torch.no_grad():
        samples, process = sample_t2i_ddpm(
            model=model,
            prompts=[args.prompt],
            image_size=image_size,
            num_timesteps=num_timesteps,
            device=device,
            cfg_scale=args.cfg_scale,
            num_samples_per_prompt=args.num_samples,
            process_samples=min(8, args.num_samples)
        )

    # 保存生成的图片网格
    save_image_grid(samples, os.path.join(paths["samples"], "t2i_diffusion_samples.png"), nrow=4)
    save_process_grid(process, os.path.join(paths["samples"], "t2i_diffusion_denoising_process.png"))
    print(f"采样成功！结果已保存到 {paths['samples']}")

if __name__ == "__main__":
    main()
