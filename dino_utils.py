"""Shared helpers: load a frozen DINOv2 backbone and turn images into features."""

import torch
from PIL import Image
from torchvision import transforms

PATCH_SIZE = 14  # DINOv2 ViTs cut images into 14x14 pixel patches

# Small -> big. ViT-S is fast enough on a laptop CPU; ViT-B is a good GPU default.
MODELS = {
    "small": "dinov2_vits14",  # 21M params, 384-dim features
    "base": "dinov2_vitb14",   # 86M params, 768-dim features
    "large": "dinov2_vitl14",  # 300M params, 1024-dim features
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def load_dino(size="small", device=None):
    """Download (first run only) and return a frozen DINOv2 model in eval mode."""
    device = device or get_device()
    model = torch.hub.load("facebookresearch/dinov2", MODELS[size])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def make_transform(img_size=224):
    """Resize + center crop to a multiple of the patch size, then normalize."""
    assert img_size % PATCH_SIZE == 0, f"img_size must be a multiple of {PATCH_SIZE}"
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path):
    return Image.open(path).convert("RGB")


@torch.no_grad()
def dino_features(model, batch):
    """Return (cls_embedding [B, D], patch_tokens [B, N, D]) for a normalized batch."""
    out = model.forward_features(batch)
    return out["x_norm_clstoken"], out["x_norm_patchtokens"]
