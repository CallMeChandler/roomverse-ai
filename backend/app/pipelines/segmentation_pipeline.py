from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMSegmentationPipeline:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vit_b",
        device: str | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at: {self.checkpoint_path}")
        print(f"[SAMSegmentationPipeline] Loading SAM model: {self.model_type}")
        print(f"[SAMSegmentationPipeline] Using device: {self.device}")
        print(f"[SAMSegmentationPipeline] Checkpoint path: {self.checkpoint_path}")

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        sam.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model = sam,
            points_per_side = 32,
            pred_iou_thresh = 0.86,
            stability_score_thresh = 0.92,
            crop_n_layers = 1,
            crop_n_points_downscale_factor = 2,
            min_mask_region_area = 100,
        )

    def predict_masks(self, image: Image.Image) -> list[dict]:
        """
        Input:
            PIL RGB image

        Returns:
            masks: list of SAM mask dictionaries
        """
        image_np = np.array(image) # RGB
        masks = self.mask_generator.generate(image_np)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        return masks
    
    @staticmethod
    def describe_masks(masks: list[dict], top_k: int =10) -> None:
        print(f"Total masks generated: {len(masks)}")
        print()

        for i, mask in enumerate(masks[:top_k]):
            bbox = mask["bbox"]  # [x_min, y_min, width, height]
            area = mask["area"]
            predicted_iou = mask.get("predicted_iou", None)
            stability_score = mask.get("stability_score", None)

            print(f"Mask {i + 1}:")
            print(f"  area            = {area}")
            print(f"  bbox            = {bbox}")
            if predicted_iou is not None:
                print(f"  predicted_iou   = {predicted_iou:.4f}")
            if stability_score is not None:
                print(f"  stability_score = {stability_score:.4f}")
            print()

    @staticmethod
    def save_mask_overlay(
        image: Image.Image,
        masks: list[dict],
        output_path: str | Path,
        top_k: int = 20,
        alpha: float=0.45,
    ) -> None:
        """
        Save original image with top-k masks overlaid in random colors.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_np = np.array(image).astype(np.uint8)
        overlay = image_np.copy()

        rng = np.random.default_rng(seed=42)

        for mask_data in masks[:top_k]:
            mask = mask_data["segmentation"]  # bool array (H, W)
            color = rng.integers(0, 255, size=3, dtype=np.uint8)

            overlay[mask] = (
                alpha * color + (1 - alpha) * overlay[mask]
            ).astype(np.uint8)

        plt.figure(figsize=(8, 10))
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    
    @staticmethod
    def save_single_mask(mask: np.ndarray, output_path: str | Path) -> None:
        """
        Save one binary mask as a black/white PNG.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mask_uint8 = (mask.astype(np.uint8)) * 255
        cv2.imwrite(str(output_path), mask_uint8)

    @staticmethod
    def save_top_masks(masks: list[dict], output_dir: str | Path, top_k: int = 5) -> None:
        """
        Save top-k binary masks separately.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, mask_data in enumerate(masks[:top_k]):
            mask = mask_data["segmentation"]
            output_path = output_dir / f"mask_{i + 1}.png"
            SAMSegmentationPipeline.save_single_mask(mask, output_path)

    @staticmethod
    def extract_mask_stats(mask_data: dict) -> dict:
        """
        Extract useful geometry/statistics from one mask.
        """
        mask = mask_data["segmentation"]
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return {
                "area": 0,
                "centroid_x": None,
                "centroid_y": None,
                "bbox": mask_data.get("bbox", None),
            }

        centroid_x = float(np.mean(x_indices))
        centroid_y = float(np.mean(y_indices))

        return {
            "area": int(mask_data["area"]),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bbox": mask_data.get("bbox", None),
        }
