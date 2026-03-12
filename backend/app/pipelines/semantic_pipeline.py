from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPSemanticPipeline:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SemanticPipeline] Loading CLIP model: {self.model_name}")
        print(f"[SemanticPipeline] Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.default_labels = [
            "wall",
            "floor",
            "bed",
            "blanket",
            "pillow",
            "curtain",
            "window",
            "lamp",
            "table",
            "chair",
            "shelf",
            "cabinet",
            "door",
            "background",
            "furniture",
            "bedroom object",
        ]

    @staticmethod
    def crop_mask_region(
        image: Image.Image,
        mask: np.ndarray,
        bbox: list[float] | None,
        pad: int = 8,
    ) -> Image.Image:
        """
        Crop the masked region using bbox, with small padding.
        Keeps original RGB values; outside-mask pixels inside crop are darkened.
        """
        image_np = np.array(image).astype(np.uint8)

        if bbox is None:
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                return image
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        else:
            x, y, w, h = bbox
            x0, y0 = int(x), int(y)
            x1, y1 = int(x + w), int(y + h)

        h_img, w_img = mask.shape
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w_img, x1 + pad)
        y1 = min(h_img, y1 + pad)

        crop_img = image_np[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]

        # Darken background outside the mask, keep mask region intact
        crop_img[~crop_mask] = (crop_img[~crop_mask] * 0.15).astype(np.uint8)

        return Image.fromarray(crop_img)

    def classify_crop(
        self,
        crop: Image.Image,
        candidate_labels: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Zero-shot classify a crop against candidate labels.
        """
        labels = candidate_labels or self.default_labels
        text_prompts = [f"a photo of {label}" for label in labels]

        inputs = self.processor(
            text=text_prompts,
            images=crop,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        ranked = sorted(
            [{"label": label, "score": float(score)} for label, score in zip(labels, probs)],
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked[:top_k]

    def label_masks(
        self,
        image: Image.Image,
        masks: list[dict],
        candidate_labels: list[str] | None = None,
        top_n_masks: int = 8,
        top_k_labels: int = 3,
    ) -> list[dict]:
        """
        Assign semantic guesses to top masks by area.
        """
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:top_n_masks]
        results = []

        for idx, mask_data in enumerate(sorted_masks, start=1):
            mask = mask_data["segmentation"].astype(bool)
            bbox = mask_data.get("bbox", None)
            crop = self.crop_mask_region(image=image, mask=mask, bbox=bbox)

            predictions = self.classify_crop(
                crop=crop,
                candidate_labels=candidate_labels,
                top_k=top_k_labels,
            )

            results.append({
                "mask_id_by_area_order": idx,
                "area": int(mask_data["area"]),
                "bbox": bbox,
                "top_predictions": [
                    {
                        "label": p["label"],
                        "score": round(p["score"], 4),
                    }
                    for p in predictions
                ],
                "best_label": predictions[0]["label"] if predictions else None,
                "best_score": round(predictions[0]["score"], 4) if predictions else None,
            })

        return results