from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class MidasDepthPipeline:
    def __init__(self, model_type: str = "MiDaS_small", device: str | None = None):
        """
        model_type options:
        - "MiDaS_small" : lighter, faster
        - "DPT_Hybrid"  : better quality, moderate
        - "DPT_Large"   : heavier, better quality
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DepthPipeline] Loading MiDaS model: {self.model_type}")
        print(f"[DepthPipeline] Using device: {self.device}")

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )

        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict_depth(self, image: Image.Image) -> np.ndarray:
        """
        Input:
            PIL RGB image

        Returns:
            depth_map: NumPy array of shape (H, W)
        """
        image_np = np.array(image)  # (H, W, C), RGB

        input_batch = self.transform(image_np).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)
        return depth_map

    @staticmethod
    def normalize_depth_for_display(
        depth_map: np.ndarray,
        invert: bool = True,
    ) -> np.ndarray:
        """
        Robust normalization for visualization.
        Uses percentiles instead of plain min/max so outliers do not kill contrast.
        """
        depth = depth_map.astype(np.float32)

        lo = np.percentile(depth, 2)
        hi = np.percentile(depth, 98)

        if hi - lo < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)

        depth = np.clip(depth, lo, hi)
        depth = (depth - lo) / (hi - lo)

        if invert:
            depth = 1.0 - depth

        return depth

    @staticmethod
    def save_depth_visualization(depth_map: np.ndarray, output_path: str | Path) -> None:
        """
        Save a colorized depth map.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth_norm = MidasDepthPipeline.normalize_depth_for_display(depth_map, invert=False)
        plt.imsave(output_path, depth_norm, cmap="plasma")

    @staticmethod
    def save_depth_grayscale(depth_map: np.ndarray, output_path: str | Path) -> None:
        """
        Save a grayscale depth image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth_norm = MidasDepthPipeline.normalize_depth_for_display(depth_map, invert=False)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), depth_uint8)

    @staticmethod
    def describe_depth(depth_map: np.ndarray) -> None:
        print("Depth map stats:")
        print(f"  shape = {depth_map.shape}")
        print(f"  dtype = {depth_map.dtype}")
        print(f"  min   = {depth_map.min():.4f}")
        print(f"  max   = {depth_map.max():.4f}")
        print(f"  mean  = {depth_map.mean():.4f}")
        print()

        print("Depth percentiles:")
        for p in [0, 1, 2, 5, 25, 50, 75, 95, 98, 99, 100]:
            print(f"  p{p:>3} = {np.percentile(depth_map, p):.4f}")
        print()