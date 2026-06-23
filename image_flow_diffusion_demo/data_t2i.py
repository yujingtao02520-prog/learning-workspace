import random
import re
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 预先定义一个包含所有可能单词的小词表，保证纯本地、零依赖运行。
WORDS = [
    "<pad>", "<unk>", "<cls>",
    # 常用描述词
    "a", "white", "drawing", "photo", "of", "the", "digit", "number",
    # 形状类
    "circle", "square", "triangle", "diamond", "line", "plus",
    # MNIST 类别
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    # FashionMNIST 类别
    "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle", "boot"
]

VOCAB: Dict[str, int] = {word: idx for idx, word in enumerate(WORDS)}
REV_VOCAB: Dict[int, str] = {idx: word for idx, word in enumerate(WORDS)}

def tokenize_prompt(prompt: str, max_length: int = 8) -> torch.Tensor:
    """
    分词器：将 prompt 字符串转换为固定长度的 Token ID 张量。
    - 首位自动添加 <cls> 用于提取全局文本特征。
    - 截断或用 <pad> 填充到 max_length。
    """
    # 替换标点符号为空格，全部转为小写
    clean_prompt = re.sub(r"[^\w\s-]", "", prompt.lower())
    words = clean_prompt.split()
    
    # 转换为 ID
    tokens = [VOCAB["<cls>"]]
    for w in words:
        if w in VOCAB:
            tokens.append(VOCAB[w])
        else:
            tokens.append(VOCAB["<unk>"])
            
    # 填充与截断
    if len(tokens) < max_length:
        tokens = tokens + [VOCAB["<pad>"]] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
        
    return torch.tensor(tokens, dtype=torch.long)

class SyntheticT2IDataset(Dataset):
    """
    文生图专用合成形状数据集。
    生成单张包含单一几何图形的二值图像，并返回对应的文本 Prompt。
    """
    def __init__(self, length: int = 50000, image_size: int = 64):
        self.length = length
        self.image_size = image_size
        self.shapes = ["circle", "square", "triangle", "diamond", "line", "plus"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        # 用 index 初始化局部随机数，保证样本可复现
        rng = random.Random(index)
        
        # 随机选择一种形状
        shape_type = rng.choice(self.shapes)
        prompt = f"a white {shape_type}"
        
        image = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(image)
        
        min_size = self.image_size // 3
        max_size = self.image_size // 2
        w = rng.randint(min_size, max_size)
        h = rng.randint(min_size, max_size)
        
        # 保证图形不会画到画布外面
        left = rng.randint(4, max(4, self.image_size - w - 4))
        top = rng.randint(4, max(4, self.image_size - h - 4))
        
        if shape_type == "circle":
            side = min(w, h)
            box = (left, top, left + side, top + side)
            draw.ellipse(box, fill=255)
        elif shape_type == "square":
            side = min(w, h)
            box = (left, top, left + side, top + side)
            draw.rectangle(box, fill=255)
        elif shape_type == "triangle":
            points = [((left + left + w) // 2, top), (left, top + h), (left + w, top + h)]
            draw.polygon(points, fill=255)
        elif shape_type == "diamond":
            points = [
                ((left + left + w) // 2, top),
                (left + w, (top + top + h) // 2),
                ((left + left + w) // 2, top + h),
                (left, (top + top + h) // 2),
            ]
            draw.polygon(points, fill=255)
        elif shape_type == "line":
            width = max(3, self.image_size // 16)
            draw.line((left, top, left + w, top + h), fill=255, width=width)
        elif shape_type == "plus":
            width = max(3, self.image_size // 16)
            cx = left + w // 2
            cy = top + h // 2
            draw.line((left, cy, left + w, cy), fill=255, width=width)
            draw.line((cx, top, cx, top + h), fill=255, width=width)

        # 随机模糊
        if rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.6)))

        arr = np.asarray(image, dtype=np.float32) / 255.0
        
        # 微小的对比度与亮度扰动
        if rng.random() < 0.5:
            arr = arr * rng.uniform(0.9, 1.1)
        arr = np.clip(arr, 0.0, 1.0)
        
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        tensor = tensor * 2.0 - 1.0  # 归一化到 [-1, 1]
        
        return tensor, prompt

class T2IDatasetWrapper(Dataset):
    """
    通用标签数据集（如 MNIST/FashionMNIST）的包装器，将整数分类标签转换为文本 Prompt。
    """
    def __init__(self, base_dataset: Dataset, prompt_map: Dict[int, str]):
        self.base_dataset = base_dataset
        self.prompt_map = prompt_map

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        img, label = self.base_dataset[index]
        prompt = self.prompt_map[label]
        return img, prompt

# 定义各个数据集的文本 prompt 映射
MNIST_PROMPT_MAP = {
    0: "the digit zero",
    1: "the digit one",
    2: "the digit two",
    3: "the digit three",
    4: "the digit four",
    5: "the digit five",
    6: "the digit six",
    7: "the digit seven",
    8: "the digit eight",
    9: "the digit nine",
}

FASHION_MNIST_PROMPT_MAP = {
    0: "a photo of a t-shirt",
    1: "a photo of a trouser",
    2: "a photo of a pullover",
    3: "a photo of a dress",
    4: "a photo of a coat",
    5: "a photo of a sandal",
    6: "a photo of a shirt",
    7: "a photo of a sneaker",
    8: "a photo of a bag",
    9: "a photo of an ankle boot",
}

def t2i_collate_fn(batch):
    """
    DataLoader 的整理函数：
    将批次数据整理成：
    1. 图像张量 [B, C, H, W]
    2. 分词后的文本 ID 张量 [B, L]
    3. 原始文本列表 List[str]
    """
    images = []
    tokenized_prompts = []
    raw_prompts = []
    for img, prompt in batch:
        images.append(img)
        tokenized_prompts.append(tokenize_prompt(prompt))
        raw_prompts.append(prompt)
        
    images = torch.stack(images, dim=0)
    tokenized_prompts = torch.stack(tokenized_prompts, dim=0)
    return images, tokenized_prompts, raw_prompts

def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def get_t2i_dataloader(
    dataset: str = "synthetic_shapes",
    batch_size: int = 128,
    image_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 4,
    dataset_size: int = 50000,
) -> DataLoader:
    """
    返回 T2I 数据加载器。返回类型为 (images, tokenized_prompts, raw_prompts)。
    """
    dataset_name = dataset.lower()
    transform = _build_transform(image_size)

    if dataset_name == "mnist":
        base_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        ds = T2IDatasetWrapper(base_ds, MNIST_PROMPT_MAP)
    elif dataset_name in {"fashion_mnist", "fashionmnist"}:
        base_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        ds = T2IDatasetWrapper(base_ds, FASHION_MNIST_PROMPT_MAP)
    elif dataset_name == "synthetic_shapes":
        ds = SyntheticT2IDataset(length=dataset_size, image_size=image_size)
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
        collate_fn=t2i_collate_fn
    )

if __name__ == "__main__":
    loader = get_t2i_dataloader(dataset="synthetic_shapes", batch_size=4, num_workers=0)
    images, tokens, prompts = next(iter(loader))
    print("Images shape:", images.shape)
    print("Tokens shape:", tokens.shape)
    print("Prompts:", prompts)
