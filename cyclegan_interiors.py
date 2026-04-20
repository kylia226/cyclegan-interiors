from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as tr


CHECKPOINT_PATH = Path(__file__).resolve().parent / "cycle_gan_wetrythebest.pt"
IMAGE_SIZE = 96

DOMAIN_STATS = {
    "A": {
        "mean": [0.4315127432346344, 0.4315127432346344, 0.4315127432346344],
        "std": [0.2660028040409088, 0.2556802034378052, 0.2579147517681122],
    },
    "B": {
        "mean": [0.4315127432346344, 0.4315127432346344, 0.3591417074203491],
        "std": [0.24365024268627167, 0.2356622815132141, 0.2485746294260025],
    },
}


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_val_transform(domain: str):
    stats = DOMAIN_STATS[domain]
    return tr.Compose(
        [
            tr.ToPILImage(),
            tr.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            tr.ToTensor(),
            tr.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )


def de_normalize(image: torch.Tensor, domain: str) -> np.ndarray:
    stats = DOMAIN_STATS[domain]
    mean_tensor = torch.tensor(stats["mean"], dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(stats["std"], dtype=image.dtype).view(-1, 1, 1)
    image = image.detach().cpu() * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, activation: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, num_res_blocks: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=7, padding=3),
            ConvBlock(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(hidden_channels * 4) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels * 8, out_channels, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class CycleGAN(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        generator_channels: int = 32,
        discriminator_channels: int = 32,
        num_res_blocks: int = 2,
        discriminator_out_channels: int = 1,
    ):
        super().__init__()
        self.generators = nn.ModuleDict(
            {
                "a_to_b": Generator(img_channels, generator_channels, num_res_blocks),
                "b_to_a": Generator(img_channels, generator_channels, num_res_blocks),
            }
        )
        self.discriminators = nn.ModuleDict(
            {
                "a": Discriminator(img_channels, discriminator_channels, discriminator_out_channels),
                "b": Discriminator(img_channels, discriminator_channels, discriminator_out_channels),
            }
        )


def create_model(device: torch.device | None = None) -> CycleGAN:
    device = device or get_device()
    return CycleGAN().to(device)


def load_checkpoint(model: CycleGAN, checkpoint_path: str | Path = CHECKPOINT_PATH, device: torch.device | None = None):
    device = device or get_device()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def preprocess_image(image: Image.Image, source_domain: str) -> torch.Tensor:
    transform = get_val_transform(source_domain)
    image_np = np.array(image.convert("RGB"))
    return transform(image_np).unsqueeze(0)


def tensor_to_pil(image_tensor: torch.Tensor, target_domain: str) -> Image.Image:
    image = de_normalize(image_tensor[0], target_domain)
    return Image.fromarray((image * 255).astype(np.uint8))
