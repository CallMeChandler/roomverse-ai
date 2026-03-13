from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GameMapPipeline:
    def __init__(self):
        self.spawn_labels = {"bed", "blanket", "pillow"}
        self.walkable_labels = {"floor"}
        self.boundary_labels = {"wall", "curtain", "window", "door"}
        self.obstacle_labels = {"chair", "table", "cabinet", "shelf", "lamp"}
        self.poi_labels = {"lamp", "window", "door", "shelf", "cabinet", "table"}

        self.role_priority = ["spawn", "obstacle", "poi", "boundary", "walkable", "unknown", "empty"]

    def classify_region_role(self, rec: dict) -> str:
        label = rec.get("semantic_best_label")
        area_label = rec.get("area_label")
        depth_label = rec.get("depth_label")
        pos_y = rec.get("position_y")
        pos_x = rec.get("position_x")

        if label in self.spawn_labels:
            return "spawn"

        if label in self.walkable_labels:
            return "walkable"

        if label in self.boundary_labels:
            return "boundary"

        if label in self.obstacle_labels:
            return "obstacle"

        if label in self.poi_labels:
            return "poi"

        # fallback room priors
        if area_label == "large" and pos_y == "top":
            return "boundary"

        if area_label == "medium" and depth_label == "near":
            return "obstacle"

        if area_label in {"medium", "large"} and pos_y == "bottom":
            return "walkable"

        if pos_x in {"left", "right"} and area_label == "medium":
            return "boundary"

        return "unknown"

    @staticmethod
    def _grid_cell_bounds(
        row: int,
        col: int,
        grid_rows: int,
        grid_cols: int,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        y0 = int(row * height / grid_rows)
        y1 = int((row + 1) * height / grid_rows)
        x0 = int(col * width / grid_cols)
        x1 = int((col + 1) * width / grid_cols)
        return y0, y1, x0, x1

    @staticmethod
    def _cell_center_label(row: int, col: int, grid_rows: int, grid_cols: int) -> str:
        y_label = "top" if row < grid_rows / 3 else "middle" if row < 2 * grid_rows / 3 else "bottom"
        x_label = "left" if col < grid_cols / 3 else "center" if col < 2 * grid_cols / 3 else "right"
        return f"{x_label}-{y_label}"

    @staticmethod
    def _role_score(role: str) -> float:
        scores = {
            "spawn": 6.0,
            "obstacle": 5.0,
            "poi": 4.0,
            "boundary": 3.0,
            "walkable": 2.0,
            "unknown": 1.0,
            "empty": 0.0,
        }
        return scores.get(role, 0.0)

    def build_symbolic_map(
        self,
        semantic_reasoning_records: list[dict],
        image_shape: tuple[int, int],
    ) -> dict:
        height, width = image_shape
        entities = []

        for rec in semantic_reasoning_records:
            role = self.classify_region_role(rec)

            entity = {
                "mask_id": rec["mask_id"],
                "semantic_label": rec.get("semantic_best_label"),
                "semantic_score": rec.get("semantic_best_score"),
                "game_role": role,
                "position_label": rec["position_label"],
                "depth_label": rec["depth_label"],
                "area_label": rec["area_label"],
                "bbox": rec["bbox"],
                "centroid_x": rec["centroid_x"],
                "centroid_y": rec["centroid_y"],
                "priority_score": rec["priority_score"],
            }
            entities.append(entity)

        # Keep only one strongest spawn candidate
        spawn_entities = [e for e in entities if e["game_role"] == "spawn"]
        if len(spawn_entities) > 1:
            best_spawn = max(spawn_entities, key=lambda x: (x["priority_score"], x.get("semantic_score") or 0.0))
            for e in entities:
                if e["game_role"] == "spawn" and e["mask_id"] != best_spawn["mask_id"]:
                    e["game_role"] = "poi"

        return {
            "image_width": width,
            "image_height": height,
            "entities": entities,
        }

    def build_grid_map(
        self,
        semantic_reasoning_records: list[dict],
        filtered_masks: list[dict],
        image_shape: tuple[int, int],
        grid_rows: int = 8,
        grid_cols: int = 8,
        overlap_threshold: float = 0.08,
    ) -> dict:
        """
        Improved grid map using mask-cell overlap, not just centroid.
        filtered_masks must be area-sorted and aligned with mask_id (1-based).
        """
        height, width = image_shape

        rec_by_mask_id = {rec["mask_id"]: rec for rec in semantic_reasoning_records}
        mask_by_mask_id = {idx: m for idx, m in enumerate(filtered_masks, start=1)}

        grid = []

        for row in range(grid_rows):
            grid_row = []
            for col in range(grid_cols):
                y0, y1, x0, x1 = self._grid_cell_bounds(
                    row=row,
                    col=col,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    height=height,
                    width=width,
                )

                cell_h = max(1, y1 - y0)
                cell_w = max(1, x1 - x0)
                cell_area = cell_h * cell_w

                cell_candidates = []

                for mask_id, rec in rec_by_mask_id.items():
                    mask_data = mask_by_mask_id.get(mask_id)
                    if mask_data is None:
                        continue

                    mask = mask_data["segmentation"].astype(bool)
                    cell_mask = mask[y0:y1, x0:x1]
                    overlap_pixels = int(cell_mask.sum())

                    if overlap_pixels <= 0:
                        continue

                    overlap_ratio = overlap_pixels / cell_area
                    role = self.classify_region_role(rec)

                    if overlap_ratio >= overlap_threshold or role == "spawn":
                        score = (
                            overlap_ratio * 5.0
                            + rec["priority_score"] * 1.5
                            + (rec.get("semantic_best_score") or 0.0) * 1.0
                            + self._role_score(role) * 0.5
                        )

                        cell_candidates.append({
                            "mask_id": mask_id,
                            "role": role,
                            "semantic_label": rec.get("semantic_best_label"),
                            "overlap_ratio": round(overlap_ratio, 4),
                            "score": round(score, 4),
                        })

                if not cell_candidates:
                    tile_type = "empty"
                    notes = []
                else:
                    cell_candidates = sorted(cell_candidates, key=lambda x: x["score"], reverse=True)
                    tile_type = cell_candidates[0]["role"]
                    notes = cell_candidates

                grid_row.append({
                    "row": row,
                    "col": col,
                    "cell_label": self._cell_center_label(row, col, grid_rows, grid_cols),
                    "tile_type": tile_type,
                    "notes": notes,
                })

            grid.append(grid_row)

        grid_map = {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid": grid,
        }

        self._cleanup_grid(grid_map)
        return grid_map

    def _cleanup_grid(self, grid_map: dict) -> None:
        """
        Light smoothing:
        - keep only strongest single spawn cluster core
        - remove isolated boundary/obstacle/poi cells
        - grow walkable slightly around existing walkable/floor-like space
        """
        grid = grid_map["grid"]
        rows = len(grid)
        cols = len(grid[0]) if rows else 0

        def neighbors(r: int, c: int):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        yield rr, cc

        # Keep only one best spawn cell
        spawn_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]["tile_type"] == "spawn":
                    best_score = max((n["score"] for n in grid[r][c]["notes"]), default=0.0)
                    spawn_cells.append((best_score, r, c))

        if len(spawn_cells) > 1:
            spawn_cells.sort(reverse=True)
            _, keep_r, keep_c = spawn_cells[0]
            for _, r, c in spawn_cells[1:]:
                grid[r][c]["tile_type"] = "poi"

        # Remove isolated hard cells
        for target in ["boundary", "obstacle", "poi"]:
            to_downgrade = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c]["tile_type"] != target:
                        continue
                    same_neighbors = sum(1 for rr, cc in neighbors(r, c) if grid[rr][cc]["tile_type"] == target)
                    if same_neighbors == 0:
                        to_downgrade.append((r, c))
            for r, c in to_downgrade:
                grid[r][c]["tile_type"] = "empty"

        # Grow walkable a little if surrounded by walkable cells
        to_walkable = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]["tile_type"] != "empty":
                    continue
                walkable_neighbors = sum(1 for rr, cc in neighbors(r, c) if grid[rr][cc]["tile_type"] == "walkable")
                if walkable_neighbors >= 3:
                    to_walkable.append((r, c))
        for r, c in to_walkable:
            grid[r][c]["tile_type"] = "walkable"

    @staticmethod
    def render_ascii_grid(grid_map: dict) -> list[str]:
        symbol_map = {
            "empty": ".",
            "walkable": "_",
            "boundary": "#",
            "obstacle": "X",
            "spawn": "S",
            "poi": "P",
            "unknown": "?",
        }

        lines = []
        for row in grid_map["grid"]:
            line = ""
            for cell in row:
                line += symbol_map.get(cell["tile_type"], "?") + " "
            lines.append(line.strip())
        return lines

    @staticmethod
    def generate_game_map_summary(symbolic_map: dict, grid_map: dict) -> list[str]:
        entities = symbolic_map["entities"]

        spawn_entities = [e for e in entities if e["game_role"] == "spawn"]
        obstacle_entities = [e for e in entities if e["game_role"] == "obstacle"]
        poi_entities = [e for e in entities if e["game_role"] == "poi"]
        boundary_entities = [e for e in entities if e["game_role"] == "boundary"]
        walkable_entities = [e for e in entities if e["game_role"] == "walkable"]

        summaries = []

        if spawn_entities:
            best_spawn = max(spawn_entities, key=lambda x: x["priority_score"])
            summaries.append(
                f"Primary spawn candidate is {best_spawn.get('semantic_label')} in the {best_spawn['position_label']} region."
            )

        if obstacle_entities:
            summaries.append(f"Detected {len(obstacle_entities)} obstacle-like regions for map blocking or cover.")

        if poi_entities:
            poi_labels = sorted({e.get("semantic_label") for e in poi_entities if e.get("semantic_label")})
            if poi_labels:
                summaries.append(f"Points of interest include: {', '.join(poi_labels)}.")

        if boundary_entities:
            summaries.append(f"Detected {len(boundary_entities)} boundary-like regions for room/map edges.")

        if walkable_entities:
            summaries.append(f"Detected {len(walkable_entities)} walkable-supporting regions.")

        non_empty_cells = sum(
            1
            for row in grid_map["grid"]
            for cell in row
            if cell["tile_type"] != "empty"
        )
        summaries.append(f"Grid abstraction contains {non_empty_cells} non-empty gameplay cells.")

        return summaries

    @staticmethod
    def save_json(data: dict, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)