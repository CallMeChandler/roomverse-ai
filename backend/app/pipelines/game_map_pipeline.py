from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GameMapPipeline:
    def __init__(self):
        self.spawn_labels = {"bed", "blanket", "pillow"}
        self.walkable_labels = {"floor"}
        self.boundary_labels = {"wall", "curtain", "window", "door", "background"}
        self.obstacle_labels = {"chair", "table", "cabinet", "shelf", "lamp", "furniture"}
        self.poi_labels = {"lamp", "window", "door", "shelf", "cabinet", "table"}

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

    def classify_region_role(self, rec: dict) -> str:
        label = rec.get("semantic_best_label")
        area_label = rec.get("area_label")
        depth_label = rec.get("depth_label")
        pos_y = rec.get("position_y")

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

        # fallback heuristics
        if area_label == "large" and pos_y == "top":
            return "boundary"

        if area_label == "medium" and depth_label == "near":
            return "obstacle"

        if area_label in {"medium", "large"} and pos_y == "bottom":
            return "walkable"

        return "unknown"

    def build_symbolic_map(
        self,
        semantic_reasoning_records: list[dict],
        image_shape: tuple[int, int],
    ) -> dict:
        """
        Convert semantic room reasoning into symbolic game entities.
        """
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

        return {
            "image_width": width,
            "image_height": height,
            "entities": entities,
        }

    def build_grid_map(
        self,
        semantic_reasoning_records: list[dict],
        image_shape: tuple[int, int],
        grid_rows: int = 8,
        grid_cols: int = 8,
    ) -> dict:
        """
        Build a coarse top-down symbolic grid.
        Each cell gets a dominant tile type based on overlapping region centroids/bboxes.
        """
        height, width = image_shape
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

                cell_entities = []

                for rec in semantic_reasoning_records:
                    cx = rec["centroid_x"]
                    cy = rec["centroid_y"]

                    if x0 <= cx < x1 and y0 <= cy < y1:
                        role = self.classify_region_role(rec)
                        cell_entities.append((role, rec))

                if not cell_entities:
                    tile_type = "empty"
                    notes = []
                else:
                    # priority order for tile type
                    role_priority = ["spawn", "obstacle", "poi", "boundary", "walkable", "unknown"]
                    found_roles = [r for r, _ in cell_entities]

                    tile_type = "unknown"
                    for candidate in role_priority:
                        if candidate in found_roles:
                            tile_type = candidate
                            break

                    notes = [
                        {
                            "mask_id": rec["mask_id"],
                            "semantic_label": rec.get("semantic_best_label"),
                            "role": role,
                        }
                        for role, rec in cell_entities
                    ]

                grid_row.append({
                    "row": row,
                    "col": col,
                    "cell_label": self._cell_center_label(row, col, grid_rows, grid_cols),
                    "tile_type": tile_type,
                    "notes": notes,
                })

            grid.append(grid_row)

        return {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid": grid,
        }

    @staticmethod
    def render_ascii_grid(grid_map: dict) -> list[str]:
        """
        Render a simple ASCII-style grid for terminal viewing.
        """
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
            summaries.append(f"Points of interest include: {', '.join(poi_labels)}.")

        if boundary_entities:
            summaries.append(f"Detected {len(boundary_entities)} boundary-like regions for room/map edges.")

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