from dataclasses import dataclass, asdict


@dataclass
class DemoConfig:
    """项目默认配置，训练脚本会用 argparse 覆盖其中的字段。"""

    dataset: str = "mnist"
    image_size: int = 64
    channels: int = 1
    batch_size: int = 128
    epochs: int = 20
    lr: float = 2e-4
    model_channels: int = 64
    num_downs: int = 3
    use_attention: bool = True
    dataset_size: int = 50000
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    num_workers: int = 4
    seed: int = 42

    diffusion_num_timesteps: int = 500
    flow_sample_steps: int = 100
    sample_every: int = 1
    save_every: int = 1
    amp: bool = True
    grad_clip: float = 1.0

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = DemoConfig()
