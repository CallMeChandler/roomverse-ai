from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

class GameMapRenderer:
    def __init__(
            self,
            tile_size: int = 64,
            margin: int = 16,
            show_grid_lines: bool = True,
            show_symbols: bool = True,
    ):
        self.tile_size = tile_size
        self.margin = margin
        self.show_grid_lines = show_grid_lines
        self.show_symbols = show_symbols

        #Tile Colors(RGB)
        self.tile_colors = {
            "empty": (30, 30, 35),
            "walkable": (90, 140, 90),
            "boundary": (80, 80, 110),
            "obstacle": (130, 80, 80),
            "spawn": (220, 200, 80),
            "poi": (90, 170, 200),
            "unknown": (120, 120, 120),
        }

        self.title_symbols = {
            "empty": ".",
            "walkable": "_",
            "boundary": "#",
            "obstacle": "X",
            "spawn": "S",
            "poi": "P",
            "unknown": "?",
        }

    def render_grid_map(self, grid_map: dict) -> Image.Image:
        grid = grid_map["grid"]
        rows = len(grid)
        cols = len(grid[0]) if rows else 0

        width = cols * self.tile_size + 2 * self.margin
        height = rows * self.tile_size + 2 * self.margin

        image = Image.new("RGB", (width, height), color=(18, 18, 22))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, self.tile_size // 3))
        except Exception:
            font = ImageFont.load_default()

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                tile_type = cell["tile_type"]

                x0 = self.margin + c * self.tile_size
                y0 = self.margin + r * self.tile_size
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size

                fill = self.tile_colors.get(tile_type, self.tile_colors["unknown"])
                draw.rectangle([x0, y0, x1, y1], fill=fill)

                if self.show_grid_lines:
                    draw.rectangle([x0, y0, x1, y1], outline=(25, 25, 25), width=1)

                if self.show_symbols:
                    symbol = self.title_symbols.get(tile_type, "?")
                    bbox = draw.textbbox((0, 0), symbol, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    tx = x0 + (self.tile_size - text_w) / 2
                    ty = y0 + (self.tile_size - text_h) / 2 - 2

                    text_color = (240, 240, 240)
                    draw.text((tx, ty), symbol, fill=text_color, font=font)

        return image

    def save_grid_map(self, grid_map: dict, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.render_grid_map(grid_map)
        image.save(output_path)

    def render_legend(self) -> Image.Image:
        entries = [
            ("empty", "."),
            ("walkable", "_"),
            ("boundary", "#"),
            ("obstacle", "X"),
            ("spawn", "S"),
            ("poi", "P"),
            ("unknown", "?"),
        ]

        row_h = 42
        swatch = 24
        padding = 14
        width = 360
        height = padding * 2 + len(entries) * row_h

        image = Image.new("RGB", (width, height), color=(18, 18, 22))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()

        for i, (name, sym) in enumerate(entries):
            y = padding + i * row_h

            color = self.tile_colors.get(name, (120, 120, 120))
            draw.rectangle([padding, y, padding + swatch, y + swatch], fill=color)

            label = f"{sym}  {name}"
            draw.text((padding + swatch + 14, y + 2), label, fill=(235, 235, 235), font=font)

        return image

    def save_legend(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.render_legend()
        image.save(output_path)