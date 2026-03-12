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
from app.pipelines.reasoning_pipeline import RoomReasoningPipeline
from app.pipelines.semantic_pipeline import CLIPSemanticPipeline
from app.pipelines.game_map_pipeline import GameMapPipeline


def main() -> None:
    image_path = Path("data/input/room.jpg")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 6: Game map abstraction ---\n")

    # -------------------------------------------------
    # Part A: input recap
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
    # Part B: depth
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
    # Part C: SAM
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
    # Part D: reasoning
    # -------------------------------------------------
    reasoning_pipeline = RoomReasoningPipeline(
        min_mask_area=1500,
        max_masks=15,
    )

    reasoning_records = reasoning_pipeline.analyze_masks(
        masks=masks,
        depth_map=depth_map,
        near_is_smaller=True,
    )

    reasoning_pipeline.describe_reasoning(reasoning_records, top_k=10)

    reasoning_json_path = output_dir / "room_reasoning.json"
    reasoning_pipeline.save_reasoning_json(reasoning_records, reasoning_json_path)

    print("Saved reasoning output:")
    print(f"  reasoning json -> {reasoning_json_path}")
    print()

    # -------------------------------------------------
    # Part E: semantics with CLIP
    # -------------------------------------------------
    semantic_pipeline = CLIPSemanticPipeline()

    candidate_labels = [
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
        "door",
        "shelf",
        "cabinet",
    ]

    semantic_results = semantic_pipeline.label_masks(
        image=image,
        masks=reasoning_pipeline.filter_masks(masks),
        candidate_labels=candidate_labels,
        top_n_masks=8,
        top_k_labels=3,
    )

    semantic_reasoning_records = reasoning_pipeline.merge_semantic_labels(
        reasoning_records=reasoning_records,
        semantic_results=semantic_results,
    )

    semantic_reasoning_json_path = output_dir / "room_semantic_reasoning.json"
    reasoning_pipeline.save_semantic_reasoning_json(
        semantic_reasoning_records,
        semantic_reasoning_json_path,
    )
    reasoning_pipeline.describe_semantic_reasoning(
        semantic_reasoning_records,
        top_k=8,
    )

    print("Saved semantic reasoning output:")
    print(f"  semantic reasoning json -> {semantic_reasoning_json_path}")
    print()

    summaries = reasoning_pipeline.generate_room_summary(
        semantic_reasoning_records,
        top_k=5,
    )

    print("Semantic room summaries:")
    for line in summaries:
        print(f"  - {line}")
    print()

    # -------------------------------------------------
    # Part F: game map abstraction
    # -------------------------------------------------
    game_map_pipeline = GameMapPipeline()

    symbolic_map = game_map_pipeline.build_symbolic_map(
        semantic_reasoning_records=semantic_reasoning_records,
        image_shape=depth_map.shape,
    )

    grid_map = game_map_pipeline.build_grid_map(
        semantic_reasoning_records=semantic_reasoning_records,
        image_shape=depth_map.shape,
        grid_rows=8,
        grid_cols=8,
    )

    symbolic_map_path = output_dir / "room_symbolic_game_map.json"
    grid_map_path = output_dir / "room_grid_game_map.json"

    game_map_pipeline.save_json(symbolic_map, symbolic_map_path)
    game_map_pipeline.save_json(grid_map, grid_map_path)

    print("Saved game map outputs:")
    print(f"  symbolic map json -> {symbolic_map_path}")
    print(f"  grid map json     -> {grid_map_path}")
    print()

    ascii_lines = game_map_pipeline.render_ascii_grid(grid_map)
    print("ASCII game-map preview:")
    for line in ascii_lines:
        print(f"  {line}")
    print()

    game_summaries = game_map_pipeline.generate_game_map_summary(symbolic_map, grid_map)
    print("Game map summaries:")
    for line in game_summaries:
        print(f"  - {line}")
    print()

    print("Phase 6 success: room understanding now maps into symbolic game-map structure.\n")


if __name__ == "__main__":
    main()