from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RoomReasoningPipeline:
    def __init__(
        self,
        min_mask_area: int = 1500,
        max_masks: int = 15,
    ):
        self.min_mask_area = min_mask_area
        self.max_masks = max_masks

    def filter_masks(self, masks: list[dict]) -> list[dict]:
        filtered = [m for m in masks if int(m["area"]) >= self.min_mask_area]
        filtered = sorted(filtered, key=lambda x: x["area"], reverse=True)
        return filtered[: self.max_masks]

    @staticmethod
    def _safe_percentile(values: np.ndarray, q: float) -> float | None:
        if values.size == 0:
            return None
        return float(np.percentile(values, q))

    @staticmethod
    def _position_bucket_x(cx: float, width: int) -> str:
        ratio = cx / width
        if ratio < 1 / 3:
            return "left"
        if ratio < 2 / 3:
            return "center"
        return "right"

    @staticmethod
    def _position_bucket_y(cy: float, height: int) -> str:
        ratio = cy / height
        if ratio < 1 / 3:
            return "top"
        if ratio < 2 / 3:
            return "middle"
        return "bottom"

    @staticmethod
    def _area_bucket(area: int, image_area: int) -> str:
        ratio = area / image_area
        if ratio < 0.03:
            return "small"
        if ratio < 0.12:
            return "medium"
        return "large"

    @staticmethod
    def _compute_mask_centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None, None
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def _depth_label_from_percentile(
        depth_value: float,
        near_thresh: float,
        far_thresh: float,
        near_is_smaller: bool,
    ) -> str:
        if near_is_smaller:
            if depth_value <= near_thresh:
                return "near"
            if depth_value >= far_thresh:
                return "far"
            return "mid"
        else:
            if depth_value >= near_thresh:
                return "near"
            if depth_value <= far_thresh:
                return "far"
            return "mid"

    @staticmethod
    def _bbox_shape_label(bbox: list[float] | None) -> str:
        """
        bbox format from SAM is [x, y, w, h]
        """
        if bbox is None:
            return "unknown"

        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            return "unknown"

        ratio = h / w
        if ratio > 1.8:
            return "tall"
        if ratio < 0.6:
            return "wide"
        return "balanced"

    def analyze_masks(
        self,
        masks: list[dict],
        depth_map: np.ndarray,
        near_is_smaller: bool = True,
    ) -> list[dict]:
        height, width = depth_map.shape
        image_area = height * width

        filtered_masks = self.filter_masks(masks)

        global_p25 = float(np.percentile(depth_map, 25))
        global_p75 = float(np.percentile(depth_map, 75))

        if near_is_smaller:
            near_thresh = global_p25
            far_thresh = global_p75
        else:
            near_thresh = global_p75
            far_thresh = global_p25

        results = []

        for idx, mask_data in enumerate(filtered_masks, start=1):
            mask = mask_data["segmentation"].astype(bool)
            area = int(mask_data["area"])
            bbox = mask_data.get("bbox", None)

            cx, cy = self._compute_mask_centroid(mask)
            if cx is None or cy is None:
                continue

            mask_depth_values = depth_map[mask]
            mean_depth = float(np.mean(mask_depth_values))
            median_depth = float(np.median(mask_depth_values))
            p10_depth = self._safe_percentile(mask_depth_values, 10)
            p90_depth = self._safe_percentile(mask_depth_values, 90)

            x_pos = self._position_bucket_x(cx, width)
            y_pos = self._position_bucket_y(cy, height)
            area_label = self._area_bucket(area, image_area)
            depth_label = self._depth_label_from_percentile(
                depth_value=median_depth,
                near_thresh=near_thresh,
                far_thresh=far_thresh,
                near_is_smaller=near_is_smaller,
            )
            shape_label = self._bbox_shape_label(bbox)

            area_ratio = area / image_area
            priority_score = float(area_ratio * 0.7)

            if depth_label == "near":
                priority_score += 0.2
            elif depth_label == "mid":
                priority_score += 0.1

            result = {
                "mask_id": idx,
                "area": area,
                "area_ratio": round(area_ratio, 4),
                "area_label": area_label,
                "centroid_x": round(cx, 2),
                "centroid_y": round(cy, 2),
                "position_x": x_pos,
                "position_y": y_pos,
                "position_label": f"{x_pos}-{y_pos}",
                "bbox": bbox,
                "shape_label": shape_label,
                "depth_mean": round(mean_depth, 4),
                "depth_median": round(median_depth, 4),
                "depth_p10": round(p10_depth, 4) if p10_depth is not None else None,
                "depth_p90": round(p90_depth, 4) if p90_depth is not None else None,
                "depth_label": depth_label,
                "predicted_iou": round(float(mask_data.get("predicted_iou", 0.0)), 4),
                "stability_score": round(float(mask_data.get("stability_score", 0.0)), 4),
                "priority_score": round(priority_score, 4),
            }
            results.append(result)

        results = sorted(results, key=lambda x: x["priority_score"], reverse=True)
        return results

    @staticmethod
    def describe_reasoning(records: list[dict], top_k: int = 10) -> None:
        print(f"Structured mask records: {len(records)}")
        print()

        for rec in records[:top_k]:
            print(f"Mask {rec['mask_id']}:")
            print(f"  area            = {rec['area']} ({rec['area_label']})")
            print(f"  area_ratio      = {rec['area_ratio']}")
            print(f"  position        = {rec['position_label']}")
            print(f"  centroid        = ({rec['centroid_x']}, {rec['centroid_y']})")
            print(f"  bbox            = {rec['bbox']}")
            print(f"  shape_label     = {rec['shape_label']}")
            print(f"  depth_mean      = {rec['depth_mean']}")
            print(f"  depth_median    = {rec['depth_median']}")
            print(f"  depth_label     = {rec['depth_label']}")
            print(f"  predicted_iou   = {rec['predicted_iou']}")
            print(f"  stability_score = {rec['stability_score']}")
            print(f"  priority_score  = {rec['priority_score']}")
            print()

    @staticmethod
    def save_reasoning_json(records: list[dict], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    @staticmethod
    def interpret_room_regions(records: list[dict]) -> list[dict]:
        """
        Heuristic room-specific interpretation layer.
        Adds a probable role to each important region.
        """
        interpreted: list[dict] = []

        for rec in records:
            role = "generic region"
            confidence = "low"

            area_label = rec["area_label"]
            pos_x = rec["position_x"]
            pos_y = rec["position_y"]
            depth = rec["depth_label"]
            shape = rec["shape_label"]
            area_ratio = rec["area_ratio"]

            # Large upper / far-ish region -> likely wall/background
            if area_label == "large" and pos_y == "top":
                role = "likely background wall or upper room plane"
                confidence = "medium"

            # Large lower near/mid region -> likely bed / foreground furniture surface
            elif area_label == "large" and pos_y == "bottom" and depth in {"near", "mid"}:
                role = "likely bed, blanket, or large foreground furniture surface"
                confidence = "medium"

            # Large center-middle region -> dominant room structure / furniture mass
            elif area_label == "large" and pos_y == "middle":
                role = "likely dominant furniture or central room structure"
                confidence = "medium"

            # Tall side region -> edge wall / curtain / window-side strip
            elif shape == "tall" and pos_x in {"left", "right"}:
                role = "likely side wall edge, curtain, or vertical room-side region"
                confidence = "medium"

            # Medium near middle/side -> object cluster
            elif area_label == "medium" and depth == "near" and pos_y in {"middle", "bottom"}:
                role = "likely foreground furniture or object cluster"
                confidence = "medium"

            # Small near middle -> local object
            elif area_label == "small" and depth == "near":
                role = "likely smaller nearby object"
                confidence = "low"

            # Mid/far upper-middle -> background object or wall-attached region
            elif depth in {"mid", "far"} and pos_y in {"top", "middle"}:
                role = "likely background object or wall-adjacent region"
                confidence = "low"

            interpreted.append({
                **rec,
                "interpreted_role": role,
                "interpretation_confidence": confidence,
            })

        return interpreted

    @staticmethod
    def generate_room_summary(records: list[dict], top_k: int = 5) -> list[str]:
        summaries: list[str] = []

        interpreted = RoomReasoningPipeline.interpret_room_regions(records)

        for rec in interpreted[:top_k]:
            line = (
                f"Mask {rec['mask_id']} is a {rec['area_label']} region in the "
                f"{rec['position_label']} area, with {rec['depth_label']} depth, "
                f"and is interpreted as {rec['interpreted_role']}."
            )
            summaries.append(line)

        return summaries

    @staticmethod
    def describe_interpreted_regions(records: list[dict], top_k: int = 10) -> None:
        interpreted = RoomReasoningPipeline.interpret_room_regions(records)

        print("Interpreted room regions:")
        print()

        for rec in interpreted[:top_k]:
            print(f"Mask {rec['mask_id']}:")
            print(f"  position_label            = {rec['position_label']}")
            print(f"  area_label                = {rec['area_label']}")
            print(f"  depth_label               = {rec['depth_label']}")
            print(f"  shape_label               = {rec['shape_label']}")
            print(f"  interpreted_role          = {rec['interpreted_role']}")
            print(f"  interpretation_confidence = {rec['interpretation_confidence']}")
            print()

    @staticmethod
    def save_interpreted_json(records: list[dict], output_path: str | Path) -> None:
        interpreted = RoomReasoningPipeline.interpret_room_regions(records)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(interpreted, f, indent=2)