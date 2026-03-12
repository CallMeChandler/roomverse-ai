from pathlib import Path

from app.pipelines.image_loader import load_image, get_image_info
from app.pipelines.preprocess import (
    pil_to_numpy,
    numpy_to_chw_tensor,
    add_batch_dimension,
    normalize_tensor,
    describe_tensor,
)
from app.pipelines.depth_pipeline import MidasDepthPipeline
from app.pipelines.segmentation_pipeline import SAMSegmentationPipeline


def main() -> None:
    image_path = Path("data/input/room.jpg")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 3: Room image -> MiDaS depth + SAM segmentation ---\n")

    # -------------------------------------------------
    # Part A: Phase 1 pipeline recap
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
    print(f"  shape = {image_np.shape}")
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
    # Part B: MiDaS depth
    # -------------------------------------------------
    depth_pipeline = MidasDepthPipeline(model_type="MiDaS_small")
    depth_map = depth_pipeline.predict_depth(image)
    depth_pipeline.describe_depth(depth_map)

    color_depth_path = output_dir / "room_depth_plasma.png"
    gray_depth_path = output_dir / "room_depth_gray.png"

    depth_pipeline.save_depth_visualization(depth_map, color_depth_path)
    depth_pipeline.save_depth_grayscale(depth_map, gray_depth_path)

    print("Saved depth outputs:")
    print(f"  color depth map -> {color_depth_path}")
    print(f"  grayscale depth -> {gray_depth_path}")
    print()

    # -------------------------------------------------
    # Part C: SAM segmentation
    # -------------------------------------------------
    sam_checkpoint = Path("checkpoints/sam_vit_b_01ec64.pth")
    segmentation_pipeline = SAMSegmentationPipeline(
        checkpoint_path=sam_checkpoint,
        model_type="vit_b",
    )

    masks = segmentation_pipeline.predict_masks(image)
    segmentation_pipeline.describe_masks(masks, top_k=10)

    mask_overlay_path = output_dir / "room_sam_overlay.png"
    top_masks_dir = output_dir / "sam_top_masks"

    segmentation_pipeline.save_mask_overlay(
        image=image,
        masks=masks,
        output_path=mask_overlay_path,
        top_k=20,
        alpha=0.45,
    )
    segmentation_pipeline.save_top_masks(
        masks=masks,
        output_dir=top_masks_dir,
        top_k=5,
    )

    print("Saved segmentation outputs:")
    print(f"  mask overlay -> {mask_overlay_path}")
    print(f"  top masks dir -> {top_masks_dir}")
    print()

    # -------------------------------------------------
    # Part D: tiny geometry preview for top 3 masks
    # -------------------------------------------------
    print("Top 3 mask geometry summaries:")
    for i, mask_data in enumerate(masks[:3]):
        stats = segmentation_pipeline.extract_mask_stats(mask_data)
        print(f"Mask {i + 1}:")
        print(f"  area       = {stats['area']}")
        print(f"  centroid_x = {stats['centroid_x']}")
        print(f"  centroid_y = {stats['centroid_y']}")
        print(f"  bbox       = {stats['bbox']}")
        print()

    print("Phase 3 success: room image now produces masks + overlays.\n")


if __name__ == "__main__":
    main()