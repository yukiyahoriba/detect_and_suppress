"""Legacy patch rendering and episode-fixed geometry (unchanged computation)."""

from __future__ import annotations


from dataclasses import dataclass


import math


from pathlib import Path


import numpy as np


@dataclass(frozen=True)
class OverlaySpec:
    mode: str = "always"
    patch_path: Path = Path("patch.png")
    seed: int = 0
    short_windows: int = 3
    short_window_frames: int = 10
    non_overlapping_windows: bool = False
    min_scale: float = 0.61
    max_scale: float = 1.39
    max_rotation_deg: float = 30.0
    max_shear: float = 0.2
    no_geometry_prob: float = 0.2


def overlay_active_timesteps(length: int, spec: OverlaySpec, episode_id: int = 0) -> set[int] | None:
    if spec.mode == "none":
        return set()
    if spec.mode == "always":
        return None
    if spec.mode != "bursts":
        raise ValueError(f"Unknown overlay mode: {spec.mode}")
    window = max(1, int(spec.short_window_frames))
    n_windows = max(1, int(spec.short_windows))
    if length <= 0:
        return set()
    if length <= window:
        return set(range(length))
    rng = np.random.default_rng(spec.seed + 100003 * length + 1_000_003 * int(episode_id))
    if spec.non_overlapping_windows:
        required = n_windows * window + (n_windows - 1)
        if required > length:
            raise ValueError(
                f"Cannot place {n_windows} non-overlapping {window}-frame windows in an episode of length {length}"
            )
        slack = length - required
        extra_gaps = rng.multinomial(slack, np.full((n_windows + 1,), 1.0 / (n_windows + 1)))
        starts = []
        cursor = int(extra_gaps[0])
        for index in range(n_windows):
            starts.append(cursor)
            cursor += window
            if index + 1 < n_windows:
                cursor += 1 + int(extra_gaps[index + 1])
    else:
        starts = rng.choice(
            np.arange(0, length - window + 1),
            size=min(n_windows, length - window + 1),
            replace=False,
        ).tolist()
    active: set[int] = set()
    for start in starts:
        active.update(range(int(start), min(length, int(start) + window)))
    return active


def should_overlay(local_index: int, active_timesteps: set[int] | None) -> bool:
    return active_timesteps is None or int(local_index) in active_timesteps


def overlay_patch_on_image(image, patch_image, *, spec: OverlaySpec, episode_id: int, timestep: int, camera_index: int):
    from PIL import Image

    if isinstance(image, Image.Image):
        base = image.convert("RGBA")
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        base = Image.fromarray(arr).convert("RGBA")

    patch = patch_image.convert("RGBA")
    width, height = base.size
    # Keep the spatial placement fixed within an episode. Agent and wrist cameras
    # intentionally share the same sampled geometry. The temporal schedule decides
    # whether this already-sampled patch is visible.
    _ = camera_index
    rng = np.random.default_rng(spec.seed + int(episode_id) * 1_000_003)

    scale = float(rng.uniform(spec.min_scale, spec.max_scale))
    patch_w = max(1, min(width, int(round(patch.width * scale))))
    patch_h = max(1, min(height, int(round(patch.height * scale))))
    patch = patch.resize((patch_w, patch_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    max_x = max(0, width - patch_w)
    max_y = max(0, height - patch_h)
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    canvas.alpha_composite(patch, (x, y))

    if float(rng.random()) >= spec.no_geometry_prob:
        angle = math.radians(float(rng.uniform(-spec.max_rotation_deg, spec.max_rotation_deg)))
        shx = float(rng.uniform(-spec.max_shear, spec.max_shear))
        shy = float(rng.uniform(-spec.max_shear, spec.max_shear))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.array(
            [
                [1.0, shx, 0.0],
                [shy, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        center_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        origin_to_center = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        forward = origin_to_center @ matrix @ center_to_origin
        inverse = np.linalg.inv(forward)
        coeffs = tuple(float(x) for x in inverse[:2, :].reshape(-1))
        canvas = canvas.transform(
            (width, height),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )

    out = Image.alpha_composite(base, canvas).convert("RGB")
    return out


