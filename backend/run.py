from pathlib import Path

from app.pipelines.image_loader import load_image, get_image_info
from app.pipelines.interior_pipeline import InteriorDesignPipeline


def main() -> None:
    image_path = Path("data/input/room.jpg")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 8A: Interior redesign MVP ---\n")

    # -------------------------------------------------
    # Part A: load room image
    # -------------------------------------------------
    image = load_image(image_path)
    info = get_image_info(image)

    print("Loaded image info:")
    print(f"  width  = {info['width']}")
    print(f"  height = {info['height']}")
    print(f"  mode   = {info['mode']}")
    print()

    # -------------------------------------------------
    # Part B: choose style
    # -------------------------------------------------
    selected_style = "minimalist"
    # try later:
    # "luxury"
    # "cyberpunk"
    # "japanese_zen"

    print(f"Selected style: {selected_style}")
    print()

    # -------------------------------------------------
    # Part C: generate redesign
    # -------------------------------------------------
    interior_pipeline = InteriorDesignPipeline()

    redesigned_image, used_prompt = interior_pipeline.generate_redesign(
        image=image,
        style_name=selected_style,
        strength=0.42,            # lower = preserve layout more
        guidance_scale=7.0,
        num_inference_steps=12,   # CPU-friendly
        target_size=384,          # CPU-friendly
    )

    print("Used prompt:")
    print(f"  {used_prompt}")
    print()

    # -------------------------------------------------
    # Part D: save outputs
    # -------------------------------------------------
    prepared_input = interior_pipeline.prepare_init_image(image, target_size=384)

    input_path = output_dir / f"interior_input_{selected_style}.png"
    redesigned_path = output_dir / f"interior_redesign_{selected_style}.png"
    compare_path = output_dir / f"interior_compare_{selected_style}.png"

    interior_pipeline.save_image(prepared_input, input_path)
    interior_pipeline.save_image(redesigned_image, redesigned_path)
    interior_pipeline.save_side_by_side(prepared_input, redesigned_image, compare_path)

    print("Saved interior mode outputs:")
    print(f"  prepared input -> {input_path}")
    print(f"  redesigned     -> {redesigned_path}")
    print(f"  comparison     -> {compare_path}")
    print()

    print("Phase 8A success: room image now produces a styled interior redesign.\n")


if __name__ == "__main__":
    main()