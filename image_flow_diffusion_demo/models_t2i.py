import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """
    将标量时间步 t 编码为 sinusoidal embedding 向量。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t 应该是 [B] 形状的张量
        if t.dim() == 0:
            t = t[None]
        # 放大时间特征，增强正弦编码区分度
        t = t.float().reshape(-1) * 1000.0

        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=t.device) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class TextEncoder(nn.Module):
    """
    轻量、可训练的文本编码器。
    将 Token IDs [B, L] 转换为文本嵌入特征 [B, L, D]。
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 20, hidden_size))  # 支持最长 20 个 token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, l = tokens.shape
        x = self.embedding(tokens)  # [B, L, D]
        x = x + self.pos_embed[:, :l]
        x = self.transformer(x)     # [B, L, D]
        return x

class PatchEmbedding(nn.Module):
    """
    图像 Patch 化投影层。
    将 [B, C, H, W] 的图像分割并投射为 [B, N, D] 的特征，并加入 2D 位置嵌入。
    """
    def __init__(self, in_channels: int, patch_size: int, hidden_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
        return x

class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) 块。
    包含自注意力层（Image Self-Attention）、交叉注意力层（Text Cross-Attention）和 MLP 块，
    全部通过时间嵌入来做自适应层归一化（AdaLN-Zero 调制）。
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 每个 block 包含 9 个调制参数（自注意力 3 个，交叉注意力 3 个，MLP 3 个）：
        # scale1, shift1, gate1, scale2, shift2, gate2, scale3, shift3, gate3
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)  # [B, 9 * D]
        scale1, shift1, gate1, scale2, shift2, gate2, scale3, shift3, gate3 = mod.chunk(9, dim=-1)

        # 1. 图像自注意力
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate1.unsqueeze(1) * h

        # 2. 文本交叉注意力（Image queries Text keys/values）
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.cross_attn(h, context, context, need_weights=False)[0]
        x = x + gate2.unsqueeze(1) * h

        # 3. 前馈网络 MLP
        h = self.norm3(x)
        h = h * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate3.unsqueeze(1) * h

        return x

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 顶层模型。
    可作为扩散模型的噪声预测器 (epsilon_theta) 或是 Flow Matching 的速度场预测器 (v_theta)。
    """
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        vocab_size: int = 50,
        image_size: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.hidden_size = hidden_size

        self.text_encoder = TextEncoder(vocab_size, hidden_size, num_layers=2, num_heads=num_heads)
        self.patch_embed = PatchEmbedding(in_channels, patch_size, hidden_size, image_size)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终层归一化与调制
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2)
        )
        self.linear_out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化权重。特别使用 AdaLN-Zero 方法：
        将用于缩放和出门（gate）的参数初始化为 0，这使网络各块在初始化时等价于恒等映射。
        """
        # 1. 2D 位置嵌入初始化为标准高斯分布
        nn.init.normal_(self.patch_embed.pos_embed, std=0.02)
        nn.init.normal_(self.text_encoder.pos_embed, std=0.02)

        # 2. Linear 层和 Conv2d 层采用 Xavier 初始化
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # 3. 将所有 AdaLN 的最后一层 Linear 的权重与偏置初始化为 0 (AdaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)
        
        # 4. 将输出投影层初始化为 0
        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [B, N, P*P*C] 的 patch 序列重新排列为图像 [B, C, H, W]
        """
        p = self.patch_size
        g = self.grid_size
        b = x.shape[0]

        # x: [B, g*g, p*p*C]
        x = x.reshape(b, g, g, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, g, p, g, p]
        x = x.reshape(b, self.in_channels, g * p, g * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] (加噪图像)
        t: [B] (时间步)
        text_tokens: [B, L] (文本 Token IDs)
        """
        # 1. 文本编码
        context = self.text_encoder(text_tokens)  # [B, L, D]

        # 2. 图像 patch 嵌入
        h = self.patch_embed(x)  # [B, N, D]

        # 3. 时间嵌入
        t_emb = self.time_embed(t)  # [B, D]

        # 4. Transformer Block 堆叠
        for block in self.blocks:
            h = block(h, context, t_emb)

        # 5. 最后一层自适应调制与投影
        mod = self.adaLN_final(t_emb)  # [B, 2 * D]
        scale, shift = mod.chunk(2, dim=-1)
        h = self.norm_final(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 投影回原 patch 的像素维度
        h = self.linear_out(h)  # [B, N, P*P*C]

        # 还原为图像
        out = self.unpatchify(h)  # [B, C, H, W]
        return out

if __name__ == "__main__":
    # 测试模型前向传播
    model = DiT(in_channels=1, patch_size=4, hidden_size=128, num_layers=4, num_heads=4, vocab_size=50, image_size=64)
    x = torch.randn(2, 1, 64, 64)
    t = torch.rand(2)
    tokens = torch.randint(0, 50, (2, 8))
    out = model(x, t, tokens)
    print("Output shape:", out.shape)  # 期望 [2, 1, 64, 64]
