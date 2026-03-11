from pathlib import Path

import numpy as np

from app.pipelines.image_loader import load_image, get_image_info
from app.pipelines.preprocess import (
    pil_to_numpy,
    numpy_to_chw_tensor,
    add_batch_dimension,
    normalize_tensor,
    describe_tensor,
)
from app.pipelines.depth_pipeline import MidasDepthPipeline


def main() -> None:
    image_path = Path("data/input/room.jpg")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 2: Room image -> MiDaS depth map ---\n")

    # -------------------------------------------------
    # Part A: Reuse Phase 1 understanding pipeline
    # -------------------------------------------------
    image = load_image(image_path)
    info = get_image_info(image)

    print("Loaded image info:")
    print(f"  width  = {info['width']}")
    print(f"  height = {info['height']}")
    print(f"  mode   = {info['mode']}")
    print()

    image_np = pil_to_numpy(image)
    print("NumPy image:")
    print(f"  shape = {image_np.shape}")   # (H, W, C)
    print(f"  dtype = {image_np.dtype}")
    print(f"  min   = {image_np.min():.1f}")
    print(f"  max   = {image_np.max():.1f}")
    print(f"  top-left pixel RGB = {image_np[0, 0]}")
    print()

    image_tensor = numpy_to_chw_tensor(image_np)
    describe_tensor("Tensor after HWC -> CHW and scaling to [0,1]", image_tensor)

    batched_tensor = add_batch_dimension(image_tensor)
    describe_tensor("Tensor after batch dimension", batched_tensor)

    normalized_tensor = normalize_tensor(batched_tensor)
    describe_tensor("Tensor after normalization", normalized_tensor)

    # -------------------------------------------------
    # Part B: MiDaS depth inference
    # -------------------------------------------------
    depth_pipeline = MidasDepthPipeline(model_type="MiDaS_small")
    depth_map = depth_pipeline.predict_depth(image)
    depth_pipeline.describe_depth(depth_map)

    # -------------------------------------------------
    # Part C: Save outputs
    # -------------------------------------------------
    color_depth_path = output_dir / "room_depth_plasma.png"
    gray_depth_path = output_dir / "room_depth_gray.png"

    depth_pipeline.save_depth_visualization(depth_map, color_depth_path)
    depth_pipeline.save_depth_grayscale(depth_map, gray_depth_path)
    depth_pipeline.save_depth_grayscale(depth_map, gray_depth_path)

    print("Saved depth outputs:")
    print(f"  color depth map -> {color_depth_path}")
    print(f"  grayscale depth -> {gray_depth_path}")
    print()

    # -------------------------------------------------
    # Part D: Tiny interpretation
    # -------------------------------------------------
    h, w = depth_map.shape
    center_depth = depth_map[h // 2, w // 2]
    top_depth = depth_map[h // 4, w // 2]
    bottom_depth = depth_map[(3 * h) // 4, w // 2]

    print("Quick depth probes:")
    print(f"  top-center depth value    = {top_depth:.4f}")
    print(f"  center depth value        = {center_depth:.4f}")
    print(f"  bottom-center depth value = {bottom_depth:.4f}")
    print()
    print("These are raw relative-depth values, not exact meters.")
    print("Phase 2 success: room image now produces a dense depth map.\n")


if __name__ == "__main__":
    main()