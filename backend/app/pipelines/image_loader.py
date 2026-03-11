from pathlib import Path
from PIL import Image

def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk and convert it to RGB.

    Args:
        image_path: Path to the input image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file is not a valid image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not open image: {image_path}") from e
    
    image = image.convert("RGB")
    return image

def get_image_info(image: Image.Image) -> dict:
    """
    Return basic metadata about the PIL image. """
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "mode": image.mode,
    }
