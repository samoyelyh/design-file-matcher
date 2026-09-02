from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from .media import open_visual


class PHashProvider:
    """Stable 64-bit perceptual hash and Hamming distance implementation."""

    feature_version = "phash64_v1"

    def compute_image(self, image: Image.Image) -> str:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        low = dctn(pixels, type=2, norm="ortho")[:8, :8]
        median = float(np.median(low.flatten()[1:]))
        bits = (low > median).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:016x}"

    def compute_path(self, path: Path) -> str:
        return self.compute_image(open_visual(path))

    @staticmethod
    def distance(left: str, right: str) -> int:
        return (int(left, 16) ^ int(right, 16)).bit_count()

    def distance_matrix(self, queries: list[str], mains: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [self.distance(query, main) for main in mains]
                for query in queries
            ],
            dtype=np.int16,
        )


def source_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns
