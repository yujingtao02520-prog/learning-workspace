import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SyntheticShapesDataset(Dataset):
    """无需下载的合成形状数据集，支持更大的 64x64 训练任务。"""

    def __init__(self, length: int = 50000, image_size: int = 64):
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def _random_box(self, rng: random.Random) -> Tuple[int, int, int, int]:
        min_size = max(6, self.image_size // 8)
        max_size = max(min_size + 1, self.image_size // 2)
        w = rng.randint(min_size, max_size)
        h = rng.randint(min_size, max_size)
        left = rng.randint(2, max(2, self.image_size - w - 2))
        top = rng.randint(2, max(2, self.image_size - h - 2))
        return left, top, left + w, top + h

    def _draw_shape(self, draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
        shape_type = rng.choice(
            ["circle", "ellipse", "square", "rectangle", "triangle", "diamond", "line", "plus"]
        )
        fill = rng.randint(150, 255)
        outline = rng.randint(120, 255)
        width = rng.randint(1, max(1, self.image_size // 24))
        box = self._random_box(rng)
        left, top, right, bottom = box

        if shape_type == "circle":
            side = min(right - left, bottom - top)
            box = (left, top, left + side, top + side)
            draw.ellipse(box, fill=fill if rng.random() < 0.8 else None, outline=outline, width=width)
        elif shape_type == "ellipse":
            draw.ellipse(box, fill=fill if rng.random() < 0.8 else None, outline=outline, width=width)
        elif shape_type == "square":
            side = min(right - left, bottom - top)
            box = (left, top, left + side, top + side)
            draw.rectangle(box, fill=fill if rng.random() < 0.8 else None, outline=outline, width=width)
        elif shape_type == "rectangle":
            draw.rectangle(box, fill=fill if rng.random() < 0.8 else None, outline=outline, width=width)
        elif shape_type == "triangle":
            points = [((left + right) // 2, top), (left, bottom), (right, bottom)]
            draw.polygon(points, fill=fill)
        elif shape_type == "diamond":
            points = [
                ((left + right) // 2, top),
                (right, (top + bottom) // 2),
                ((left + right) // 2, bottom),
                (left, (top + bottom) // 2),
            ]
            draw.polygon(points, fill=fill)
        elif shape_type == "line":
            draw.line((left, top, right, bottom), fill=fill, width=max(2, width + 1))
        else:
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            half_w = max(2, (right - left) // 2)
            half_h = max(2, (bottom - top) // 2)
            draw.line((cx - half_w, cy, cx + half_w, cy), fill=fill, width=max(2, width + 1))
            draw.line((cx, cy - half_h, cx, cy + half_h), fill=fill, width=max(2, width + 1))

    def __getitem__(self, index: int):
        # 用 index 初始化局部随机数，让每个样本可复现。
        rng = random.Random(index)
        np_rng = np.random.default_rng(index)
        image = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(image)

        # 更大的任务不再只画一个形状，而是随机叠加 1-4 个形状。
        for _ in range(rng.randint(1, 4)):
            self._draw_shape(draw, rng)

        if rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

        arr = np.asarray(image, dtype=np.float32) / 255.0

        # 轻微噪声和对比度扰动能让合成数据更接近一个“小任务”而不是固定模板记忆。
        if rng.random() < 0.6:
            arr = arr * rng.uniform(0.85, 1.15)
        if rng.random() < 0.5:
            arr = arr + np_rng.normal(0.0, 0.02, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor, 0


def _build_transform(image_size: int):
    """MNIST/FashionMNIST 变换：resize、转 tensor、归一化到 [-1, 1]。"""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_dataloader(
    dataset: str = "mnist",
    batch_size: int = 128,
    image_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 4,
    dataset_size: int = 50000,
) -> DataLoader:
    """
    返回图像 shape 为 [B, 1, H, W]、像素范围为 [-1, 1] 的 dataloader。
    支持 mnist、fashion_mnist、synthetic_shapes。
    """

    dataset_name = dataset.lower()
    transform = _build_transform(image_size)

    if dataset_name == "mnist":
        ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name in {"fashion_mnist", "fashionmnist"}:
        ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "synthetic_shapes":
        ds = SyntheticShapesDataset(length=dataset_size, image_size=image_size)
    else:
        raise ValueError("dataset 必须是 mnist、fashion_mnist 或 synthetic_shapes。")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


if __name__ == "__main__":
    loader = get_dataloader(dataset="synthetic_shapes", batch_size=4, num_workers=0)
    images, _ = next(iter(loader))
    print(images.shape, images.min().item(), images.max().item())
