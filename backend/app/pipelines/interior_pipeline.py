from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image


class InteriorDesignPipeline:
    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[InteriorPipeline] Loading diffusion model: {self.model_id}")
        print(f"[InteriorPipeline] Using device: {self.device}")

        # CPU-friendly loading
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

        self.pipe = self.pipe.to(self.device)

        # Mild CPU memory help
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        self.style_presets = {
            "minimalist": (
                "a clean minimalist modern bedroom interior, soft natural lighting, "
                "neutral colors, elegant furniture, uncluttered design, realistic interior photo"
            ),
            "luxury": (
                "a luxury bedroom interior, premium materials, warm ambient lighting, "
                "high-end hotel style, elegant textures, realistic interior photo"
            ),
            "cyberpunk": (
                "a cyberpunk gaming bedroom interior, neon accent lights, futuristic setup, "
                "LED glow, moody lighting, realistic room redesign"
            ),
            "japanese_zen": (
                "a japanese zen bedroom interior, natural wood, calm minimalist decor, "
                "soft daylight, balanced layout, realistic interior photo"
            ),
        }

    def get_style_prompt(self, style_name: str) -> str:
        key = style_name.strip().lower()
        if key not in self.style_presets:
            raise ValueError(
                f"Unknown style '{style_name}'. Available styles: {list(self.style_presets.keys())}"
            )
        return self.style_presets[key]

    @staticmethod
    def prepare_init_image(image: Image.Image, target_size: int = 384) -> Image.Image:
        """
        Resize image for CPU-friendly img2img.
        Keeps aspect ratio, then center-crops to a square.
        """
        image = image.convert("RGB")

        w, h = image.size
        scale = target_size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = image.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        right = left + target_size
        bottom = top + target_size

        return image.crop((left, top, right, bottom))

    def generate_redesign(
        self,
        image: Image.Image,
        style_name: str,
        negative_prompt: str | None = None,
        strength: float = 0.42,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 12,
        target_size: int = 384,
    ) -> tuple[Image.Image, str]:
        """
        Generate a redesigned room image from an input room photo.
        """
        prompt = self.get_style_prompt(style_name)

        if negative_prompt is None:
            negative_prompt = (
                "blurry, distorted room, extra furniture limbs, warped walls, "
                "low quality, bad perspective, messy artifacts, unrealistic geometry"
            )

        init_image = self.prepare_init_image(image, target_size=target_size)

        generator = None
        if self.device == "cuda":
            generator = torch.Generator(device=self.device).manual_seed(42)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        return result, prompt

    @staticmethod
    def save_image(image: Image.Image, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    @staticmethod
    def save_side_by_side(
        original: Image.Image,
        redesigned: Image.Image,
        output_path: str | Path,
        gap: int = 16,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original = original.convert("RGB")
        redesigned = redesigned.convert("RGB")

        # match heights
        target_h = max(original.height, redesigned.height)

        def resize_to_height(img: Image.Image, h: int) -> Image.Image:
            w = int(img.width * (h / img.height))
            return img.resize((w, h), Image.LANCZOS)

        original_r = resize_to_height(original, target_h)
        redesigned_r = resize_to_height(redesigned, target_h)

        canvas_w = original_r.width + redesigned_r.width + gap
        canvas_h = target_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
        canvas.paste(original_r, (0, 0))
        canvas.paste(redesigned_r, (original_r.width + gap, 0))

        canvas.save(output_path)