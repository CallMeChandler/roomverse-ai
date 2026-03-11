import numpy as np
import torch
from PIL import Image

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to NumPy array with shape (H, W, C).
    """
    return np.array(image, dtype=np.float32)

def numpy_to_chw_tensor(image_array: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy image array from (H, W, C) to PyTorch tensor with shape (C, H, W).

    Also scales pixel values from [0, 255] to [0,1].

    Args:
        image_array: NumPy array of shape (H, W, C) 
    
    Returns:
        Tensor of shape (C, H, W), dtype float32
    """
    if image_array.ndim != 3:
        raise ValueError(f"Expected image array with 3 dimensions (H, W, C), got {image_array.shape}")
    tensor = torch.from_numpy(image_array)          # (H, W, C)
    tensor = tensor.permute(2, 0, 1)                # (C, H, W)
    tensor = tensor / 255.0                         # scale to [0, 1]
    return tensor.float()

def add_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert (C, H, W) -> (1, C, H, W)
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C, H, W), got {tuple(tensor.shape)}")
    return tensor.unsqueeze(0)


def normalize_tensor(
    tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Normalize a tensor channel-wise.

    Expects tensor shape:
    - (C, H, W) or
    - (B, C, H, W)
    """
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        mean_tensor = mean_tensor[:, None, None]
        std_tensor = std_tensor[:, None, None]
    elif tensor.ndim == 4:
        mean_tensor = mean_tensor[None, :, None, None]
        std_tensor = std_tensor[None, :, None, None]
    else:
        raise ValueError(f"Expected tensor ndim 3 or 4, got {tensor.ndim}")

    return (tensor - mean_tensor) / std_tensor


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    """
    Print useful tensor information for debugging.
    """
    print(f"{name}:")
    print(f"  shape = {tuple(tensor.shape)}")
    print(f"  dtype = {tensor.dtype}")
    print(f"  min   = {tensor.min().item():.4f}")
    print(f"  max   = {tensor.max().item():.4f}")
    print()
