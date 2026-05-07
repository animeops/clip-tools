"""Vector-stroke rasterization.

Separate from ``processing.py`` because the rendering pipeline is its own
concern: it takes raw vector binaries and brush-texture images that came out of
the chunk parser, resolves per-stroke BrushStyle parameters, and stamps pattern
images along each stroke's sampled path.

Integration point: :func:`rasterize_vectors` mutates ``clip_data`` in place,
replacing each raw-bytes vector entry with its rendered ``np.ndarray``. Call it
after ``process_chunk_binary`` and before ``process_clip_data``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clip_tools.brush_dynamics import (
    CurvePoint,
    MsvcRandom,
    apply_effector,
)
from clip_tools.constants import VectorType
from clip_tools.sqlite_records import BrushEffectorGraphDataRecord
from clip_tools.structs import process_layer_blocks
from clip_tools.structs.brush_attributes import parse_brush_pattern_image_index
from clip_tools.structs.vector import parse_vector_binary, rasterize_polylines
from clip_tools.types import BrushStyle, VectorPoint, VectorSample, VectorStroke


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Brush pattern extraction: follows BrushPatternImage → Mipmap → Offscreen chain
# -----------------------------------------------------------------------------


def extract_brush_pattern_images(
    clip_data: Dict[str, Union[Dict[int, bytes], bytes, np.ndarray]],
    dfs: Dict[str, pd.DataFrame],
) -> Dict[int, np.ndarray]:
    """Return {BrushPatternImage.MainId: grayscale_mask [0..255]}.

    Chain: ``BrushPatternImage.Mipmap`` → ``Mipmap.BaseMipmapInfo``
        → ``MipmapInfo`` (linked list of mipmap levels)
        → ``Offscreen`` → ``BlockData`` → ``clip_data`` → ``process_layer_blocks``.
    """
    result: Dict[int, np.ndarray] = {}
    if "BrushPatternImage" not in dfs or "Mipmap" not in dfs:
        return result
    for _, bpi in dfs["BrushPatternImage"].iterrows():
        pattern_id = int(bpi["MainId"])
        mm_rows = dfs["Mipmap"][dfs["Mipmap"]["MainId"] == bpi["Mipmap"]]
        if len(mm_rows) == 0:
            continue
        mm = mm_rows.iloc[0]
        top_offscreen_id = None
        cur = mm["BaseMipmapInfo"]
        while cur != 0:
            mi_rows = dfs["MipmapInfo"][dfs["MipmapInfo"]["MainId"] == cur]
            if len(mi_rows) == 0:
                break
            mi = mi_rows.iloc[0]
            if mi["ThisScale"] == 100.0:
                top_offscreen_id = mi["Offscreen"]
                break
            cur = mi["NextIndex"]
        if top_offscreen_id is None:
            continue
        off_rows = dfs["Offscreen"][dfs["Offscreen"]["MainId"] == top_offscreen_id]
        if len(off_rows) == 0:
            continue
        off = off_rows.iloc[0]
        bd = off["BlockData"].decode("ascii")
        val = clip_data.get(bd)
        if not isinstance(val, dict) or not val:
            continue
        blocks = sorted(val.items(), key=lambda x: x[0])
        try:
            arr = process_layer_blocks(blocks, off)
        except Exception as e:
            logger.debug(f"Failed to decode brush pattern {pattern_id}: {e}")
            continue
        if arr.ndim == 3:
            arr = arr[..., 3] if arr.shape[-1] == 4 else arr[..., 0]
        result[pattern_id] = arr
    return result


def get_pattern_style_images(
    dfs: Dict[str, pd.DataFrame], pattern_style_id: int
) -> List[int]:
    """Resolve BrushPatternStyle.ImageIndex (packed uint32 array) to image ids."""
    if "BrushPatternStyle" not in dfs:
        return []
    rows = dfs["BrushPatternStyle"][
        dfs["BrushPatternStyle"]["MainId"] == pattern_style_id
    ]
    if len(rows) == 0:
        return []
    return parse_brush_pattern_image_index(rows.iloc[0]["ImageIndex"])


def get_pattern_style_order(
    dfs: Dict[str, pd.DataFrame], pattern_style_id: int
) -> Tuple[int, int]:
    """Return (OrderType, Reverse2) for a BrushPatternStyle row.

    ``OrderType`` is the 6-mode pattern-cycling dispatcher
    (0=cycle, 1=ping-pong, 2=clamp, 3=random, 4=once, 5=array).
    ``Reverse2`` is a bit field: bits 0/1/2 select X-axis flip behavior
    (always / random / ping-pong); bits 4/5/6 select Y-axis flip behavior.
    """
    if "BrushPatternStyle" not in dfs:
        return 0, 0
    rows = dfs["BrushPatternStyle"][
        dfs["BrushPatternStyle"]["MainId"] == pattern_style_id
    ]
    if len(rows) == 0:
        return 0, 0
    r = rows.iloc[0]
    return int(r.get("OrderType", 0) or 0), int(r.get("Reverse2", 0) or 0)


def pattern_index_for_counter(counter: int, n: int, order_type: int) -> int:
    """Map a stamp counter to a pattern-image index given the cycling mode."""
    if n <= 0:
        return 0
    if order_type == 0:  # cycle
        return counter % n
    if order_type == 1:  # ping-pong
        if n == 1:
            return 0
        period = 2 * n - 2
        idx = counter % period
        return idx if idx < n else period - idx
    if order_type == 2:  # clamp
        return min(counter, n - 1)
    if order_type == 4:  # play-once
        return counter if counter < n else -1
    return counter % n  # 3 (random) and 5 (array) handled at call site


# -----------------------------------------------------------------------------
# Stamp primitives
# -----------------------------------------------------------------------------


def make_disc_mask(radius_px: float, hardness: float) -> np.ndarray:
    """Anti-aliased grayscale disc stamp for pens / brushes without a texture.

    Hardness in [0, 1]: 1.0 = sharp (with a 2-px AA band at the boundary,
    matching CLIP's observed output); 0.0 = fully-soft radial falloff. Inner
    ``hardness * radius`` is full-opacity, then falls off linearly to 0 at
    ``radius``.
    """
    r = max(1.5, radius_px)
    size = max(3, int(np.ceil(r * 2 + 2)))
    y, x = np.indices((size, size))
    cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
    dist = np.hypot(y - cy, x - cx)
    if hardness >= 1.0:
        mask = np.clip((r + 1.0 - dist) / 2.0, 0.0, 1.0)
    else:
        inner_r = max(0.0, hardness) * r
        outer_r = r
        mask = np.where(
            dist <= inner_r,
            1.0,
            np.clip((outer_r - dist) / max(1e-3, outer_r - inner_r), 0, 1),
        ).astype(np.float32)
    return (mask * 255).astype(np.uint8)


def disc_alpha_into(
    alpha_buf: np.ndarray,
    cx: float,
    cy: float,
    radius_x: float,
    radius_y: float,
    hardness: float,
    rotation_rad: float,
    opacity: float,
    accumulate: bool = False,
) -> None:
    """Accumulate an analytic elliptical soft stamp into an alpha buffer using
    MAX (not over-composite).

    Separate x/y radii + rotation support chisel / brush-tip brushes where
    ``ThicknessBase`` is an aspect ratio (short-axis / long-axis) and
    ``RotationBase`` sets the ellipse orientation. For round brushes, pass
    ``radius_x == radius_y`` and ``rotation_rad = 0``.

    Within a single stroke, stamps merge via MAX so overlapping AA bands
    aren't over-composited into each other (would cause periodic beading).
    """
    rx = max(0.5, radius_x)
    ry = max(0.5, radius_y)
    outer = max(rx, ry) + 2  # AA bleed + rotation slack
    x0 = int(np.floor(cx - outer))
    y0 = int(np.floor(cy - outer))
    x1 = int(np.ceil(cx + outer))
    y1 = int(np.ceil(cy + outer))
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(alpha_buf.shape[1], x1)
    dy1 = min(alpha_buf.shape[0], y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    # World-coords relative to stamp center; rotate into ellipse-local frame
    yy = np.arange(dy0, dy1) + 0.5 - cy
    xx = np.arange(dx0, dx1) + 0.5 - cx
    cos_r = float(np.cos(rotation_rad))
    sin_r = float(np.sin(rotation_rad))
    lx = xx[None, :] * cos_r + yy[:, None] * sin_r
    ly = -xx[None, :] * sin_r + yy[:, None] * cos_r
    if rx == ry:
        # Circular stamp — use simple distance-from-center in world space for
        # a clean 1-pixel AA band.
        dist = np.sqrt(lx * lx + ly * ly)
        if hardness >= 1.0:
            src_a = np.clip(rx + 0.5 - dist, 0.0, 1.0) * opacity
        else:
            inner_r = max(0.0, hardness) * rx
            src_a = (
                np.where(
                    dist <= inner_r,
                    1.0,
                    np.clip((rx - dist) / max(1e-3, rx - inner_r), 0, 1),
                )
                * opacity
            ).astype(np.float32)
    else:
        # Elliptical stamp. Use normalized ellipse distance (=1 on the
        # boundary). AA band width = 1 normalized unit / min(rx, ry), so it
        # corresponds to ~1 pixel along the ellipse's shorter axis.
        norm_dist = np.sqrt((lx / rx) ** 2 + (ly / ry) ** 2)
        aa_half = 0.5 / min(rx, ry)
        if hardness >= 1.0:
            src_a = (
                np.clip((1.0 + aa_half - norm_dist) / (2 * aa_half), 0.0, 1.0) * opacity
            )
        else:
            inner = max(0.0, hardness)
            src_a = (
                np.where(
                    norm_dist <= inner,
                    1.0,
                    np.clip((1.0 - norm_dist) / max(1e-3, 1.0 - inner), 0, 1),
                )
                * opacity
            ).astype(np.float32)
    region = alpha_buf[dy0:dy1, dx0:dx1]
    if accumulate:
        # Soft brushes: each stamp deposits `src_a` worth of ink into the
        # region, building up toward 1.0 (Porter-Duff "over" against itself).
        region[...] = region + src_a * (1.0 - region)
    else:
        # Hard brushes: overlapping stamps form a single silhouette via MAX
        # (avoids per-stamp AA boundaries in the interior).
        np.maximum(region, src_a, out=region)


def composite_alpha_onto(
    buffer: np.ndarray,
    stroke_alpha: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    """Over-composite a solid-color stroke (given by its alpha mask) onto the
    RGBA buffer in one pass. Called once per stroke after all its stamps have
    been MAX-accumulated into ``stroke_alpha``."""
    dst_a = buffer[..., 3]
    out_a = stroke_alpha + dst_a * (1.0 - stroke_alpha)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    for ch in range(3):
        buffer[..., ch] = (
            (color[ch] / 255.0) * stroke_alpha
            + buffer[..., ch] * dst_a * (1.0 - stroke_alpha)
        ) / safe_a
    buffer[..., 3] = out_a


def q15_stamp_pattern_into_alpha(
    alpha_buf: np.ndarray,
    pattern: np.ndarray,
    cx: float,
    cy: float,
    size_x: float,
    size_y: float,
    flow_q15: int,
    rotation_rad: float,
    flip_h: bool,
    flip_v: bool,
    continuous_plot: bool,
) -> None:
    """Q15 fixed-point pattern blit into a stroke-local alpha plane.

    Per-pixel formula: `pix = pix + ((target - pix) * strength) >> 15`
    (additive, default) or `cand = (target * strength) >> 15;
    if cand > pix: pix = cand` (alpha-max, when `continuous_plot` is true).
    `target = flow_q15` (per-stamp opacity in Q15); `strength` is the
    per-pixel pattern density mapped to Q15.
    """
    if size_x < 1.0 or size_y < 1.0:
        return
    bbox = int(np.ceil(max(size_x, size_y) * 1.42)) + 2
    cos_r, sin_r = float(np.cos(rotation_rad)), float(np.sin(rotation_rad))
    th, tw = pattern.shape

    x0 = int(np.floor(cx - bbox / 2.0))
    y0 = int(np.floor(cy - bbox / 2.0))
    x1, y1 = x0 + bbox, y0 + bbox
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(alpha_buf.shape[1], x1)
    dy1 = min(alpha_buf.shape[0], y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return

    wy = np.arange(dy0, dy1) + 0.5
    wx = np.arange(dx0, dx1) + 0.5
    yy, xx = np.meshgrid(wy - cy, wx - cx, indexing="ij")
    local_x = xx * cos_r + yy * sin_r
    local_y = -xx * sin_r + yy * cos_r
    u_x = local_x * (tw / size_x) + tw / 2
    u_y = local_y * (th / size_y) + th / 2
    valid = (u_x >= 0) & (u_x < tw) & (u_y >= 0) & (u_y < th)
    sx = np.clip(u_x.astype(np.int32), 0, tw - 1)
    sy = np.clip(u_y.astype(np.int32), 0, th - 1)
    if flip_h:
        sx = (tw - 1) - sx
    if flip_v:
        sy = (th - 1) - sy

    # Pattern density u8 → 15-bit coverage: `value * 0x8000 / 0xff`.
    pat_u8 = pattern[sy, sx]
    strength = (pat_u8.astype(np.int32) * 0x8000) // 255
    strength = np.where(valid, strength, 0)

    pix = alpha_buf[dy0:dy1, dx0:dx1].astype(np.int32)
    if continuous_plot:
        cand = (flow_q15 * strength) >> 15
        np.maximum(pix, cand, out=pix)
    else:
        # Only step pixels that are below the target opacity.
        active = (pix < flow_q15) & (strength > 0)
        delta = ((flow_q15 - pix) * strength) >> 15
        pix = np.where(active, pix + delta, pix)
    np.clip(pix, 0, 0x8000, out=pix)
    alpha_buf[dy0:dy1, dx0:dx1] = pix.astype(np.uint16)


def composite_q15_alpha_onto(
    buffer: np.ndarray,
    stroke_alpha_q15: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    """Composite a Q15 stroke alpha plane onto an RGBA float canvas.

    Standard "source-over" Porter-Duff composite for `CompositeMode = Normal`:
    `out_a = src_a + dst_a * (1 - src_a)`; channel `out_c = (src_c * src_a
    + dst_c * dst_a * (1 - src_a)) / out_a`. Source is straight alpha.
    """
    src_a = stroke_alpha_q15.astype(np.float32) / 32768.0
    dst_a = buffer[..., 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    for ch in range(3):
        buffer[..., ch] = (
            (color[ch] / 255.0) * src_a + buffer[..., ch] * dst_a * (1.0 - src_a)
        ) / safe_a
    buffer[..., 3] = out_a


def stamp_pattern(
    buffer: np.ndarray,
    tex_mask: np.ndarray,
    cx: float,
    cy: float,
    size_x: float,
    size_y: float,
    opacity: float,
    color: Tuple[int, int, int],
    rotation_rad: float = 0.0,
) -> None:
    """Over-composite a rotated-and-scaled grayscale mask onto an RGBA float buffer.

    ``size_x`` and ``size_y`` are the stamp's destination footprint in pixels
    *before* rotation. The texture is scaled non-uniformly so that its
    ``(tex_w, tex_h)`` maps to ``(size_x, size_y)``, then rotated. Pass equal
    values for uniform disc stamps; pass ``(size, size * tex_h / tex_w)`` to
    preserve a shaped texture's native aspect ratio.

    Subpixel-accurate: the destination grid is sampled at fractional offsets
    so small stamps align to the sub-pixel stroke path, not rounded positions.
    """
    if size_x < 1.0 or size_y < 1.0:
        return
    bbox = int(np.ceil(max(size_x, size_y) * 1.42)) + 2  # AA bleed + rotation diagonal
    cos_r, sin_r = float(np.cos(rotation_rad)), float(np.sin(rotation_rad))
    th, tw = tex_mask.shape

    x0 = int(np.floor(cx - bbox / 2.0))
    y0 = int(np.floor(cy - bbox / 2.0))
    x1, y1 = x0 + bbox, y0 + bbox
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(buffer.shape[1], x1)
    dy1 = min(buffer.shape[0], y1)
    bw, bh = dx1 - dx0, dy1 - dy0
    if bw <= 0 or bh <= 0:
        return

    wy = np.arange(dy0, dy1) + 0.5
    wx = np.arange(dx0, dx1) + 0.5
    yy, xx = np.meshgrid(wy - cy, wx - cx, indexing="ij")
    local_x = xx * cos_r + yy * sin_r
    local_y = -xx * sin_r + yy * cos_r
    u_x = local_x * (tw / size_x) + tw / 2
    u_y = local_y * (th / size_y) + th / 2
    valid = (u_x >= 0) & (u_x < tw) & (u_y >= 0) & (u_y < th)
    sx = np.clip(u_x.astype(np.int32), 0, tw - 1)
    sy = np.clip(u_y.astype(np.int32), 0, th - 1)
    src_a = (tex_mask[sy, sx].astype(np.float32) / 255.0) * opacity
    src_a = src_a * valid

    dst = buffer[dy0:dy1, dx0:dx1]
    dst_a = dst[..., 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    for ch in range(3):
        dst[..., ch] = (
            (color[ch] / 255.0) * src_a + dst[..., ch] * dst_a * (1.0 - src_a)
        ) / safe_a
    dst[..., 3] = out_a


# -----------------------------------------------------------------------------
# Vector parsing & curve sampling
# -----------------------------------------------------------------------------


CLIP_SUBPIXEL_OFFSET_AA_ON = (-0.001, -0.001)
CLIP_SUBPIXEL_OFFSET_AA_OFF = (0.499, -0.501)

# 1°-spaced sin/cos LUT for the spray scatter angle. Integer-degree
# truncation matters for sub-pixel position alignment vs the source
# renderer's stamp positions.
SIN_LUT_DEG = np.sin(np.deg2rad(np.arange(360, dtype=np.float64)))
COS_LUT_DEG = np.cos(np.deg2rad(np.arange(360, dtype=np.float64)))
"""Sub-pixel offsets applied to vertex coordinates before rasterization.

AA-off: near-half-pixel shift so integer truncation lands at pixel-center.
AA-on: essentially zero — a tiny nudge avoids exact-boundary numerical edge
cases while preserving sub-pixel precision on stroke edges."""


def standard_spline_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
    p5: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """One point on the STANDARD vector spline's segment from p2 → p3.

    6-point local interpolating spline with piecewise-quadratic basis
    weights split at t = 0.5. Interpolates p2 at t=0 and p3 at t=1; the
    outer four points only shape the curve. Divisor 99 normalizes the
    weight sum.
    """
    if t >= 0.5:
        w0 = 2 * t * t - 6 * t + 4
        w1 = -16 * t * t + 44 * t - 28
        w2 = 96 * t * t - 260 * t + 164
        w3 = -164 * t * t + 328 * t - 65
        w4 = 96 * t * t - 124 * t + 28
        w5 = -14 * t * t + 18 * t - 4
    else:
        w0 = -14 * t * t + 10 * t
        w1 = 96 * t * t - 68 * t
        w2 = -164 * t * t + 99
        w3 = 96 * t * t + 68 * t
        w4 = -16 * t * t - 12 * t
        w5 = 2 * t * t + 2 * t
    inv = 1.0 / 99.0
    x = (
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0] + w4 * p4[0] + w5 * p5[0]
    ) * inv
    y = (
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1] + w4 * p4[1] + w5 * p5[1]
    ) * inv
    return x, y


def standard_window(
    pts: List[VectorPoint], i: int
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]:
    """6-point window for the segment from pts[i] → pts[i+1].

    Endpoints are duplicated when the window walks off either end of the
    polyline (the only convention compatible with the formula's
    "interpolates p2 and p3" behavior).
    """
    n = len(pts)
    p = lambda j: (pts[max(0, min(n - 1, j))].x, pts[max(0, min(n - 1, j))].y)
    return p(i - 2), p(i - 1), p(i), p(i + 1), p(i + 2), p(i + 3)


def quadratic_bezier_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
    )


def sample_curve_points(
    pts: List[VectorPoint],
    vtype: VectorType,
    aa_on: bool = True,
) -> List[VectorSample]:
    """Walk a parsed stroke's points and return densely-interpolated samples.

    Applies the sub-pixel offset used by CLIP's vector-fill rasterizer.

    - STANDARD: 6-point local piecewise-quadratic spline (see
      `standard_spline_point`).
    - CURVE: quadratic Bezier chain. Each non-first control point owns one
      "curve" handle; the segment from control[i] → control[i+1] uses
      `(control[i], control[i+1].curve, control[i+1])`.
    - BEZIER: should be a cubic Bezier chain, but the parser currently
      captures handles only at controls i=1, i=N-2, i=N-1 (over-fit to
      3-/4-control samples). Until handle parsing is generalized for 5+
      control strokes, fall back to the STANDARD spline so paths still
      come out smooth.
    """
    dx, dy = CLIP_SUBPIXEL_OFFSET_AA_ON if aa_on else CLIP_SUBPIXEL_OFFSET_AA_OFF
    out: List[VectorSample] = []
    if len(pts) < 2:
        return out

    use_spline = vtype in (VectorType.STANDARD, VectorType.BEZIER)

    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        seg_len = max(1, int(np.hypot(p1.x - p0.x, p1.y - p0.y)))
        n = max(2, seg_len)

        window = standard_window(pts, i) if use_spline else None

        for s in range(n):
            t = s / (n - 1)
            if vtype == VectorType.CURVE and p1.curve is not None:
                x, y = quadratic_bezier_point((p0.x, p0.y), p1.curve, (p1.x, p1.y), t)
            elif window is not None:
                x, y = standard_spline_point(*window, t)
            else:
                x = p0.x + t * (p1.x - p0.x)
                y = p0.y + t * (p1.y - p0.y)
            press = p0.pressure + t * (p1.pressure - p0.pressure)
            vel = p0.velocity + t * (p1.velocity - p0.velocity)
            smo = p0.smooth + t * (p1.smooth - p0.smooth)
            tx = p0.tilt_x + t * (p1.tilt_x - p0.tilt_x)
            ang = p0.angle_deg + t * (p1.angle_deg - p0.angle_deg)
            sz = p0.size + t * (p1.size - p0.size)
            fl = p0.flow + t * (p1.flow - p0.flow)
            sop = p0.stroke_opacity + t * (p1.stroke_opacity - p0.stroke_opacity)
            # Lock bits: take the endpoint's flags directly when sampled
            # near it (lock bits don't lerp — an interior sample with a
            # locked endpoint inherits that endpoint's literal value
            # downstream).
            flags = p0.flags if t < 0.5 else p1.flags
            out.append(
                VectorSample(
                    x=x + dx,
                    y=y + dy,
                    pressure=press,
                    velocity=vel,
                    smooth=smo,
                    tilt_x=tx,
                    angle_deg=ang,
                    size=sz,
                    flow=fl,
                    stroke_opacity=sop,
                    flags=flags,
                )
            )
    return out


# -----------------------------------------------------------------------------
# Top-level renderers
# -----------------------------------------------------------------------------


ANTI_ALIAS_TO_SS = {0: 1, 1: 4, 2: 4, 3: 4}
"""CLIP's ``AntiAlias`` enum → rendering supersampling factor.

AA strength behaves as binary for vector fills: 0 renders without edge
softening; 1/2/3 all use 4× Y-supersampling plus analytic X coverage."""


def default_brush(brush_id: int) -> BrushStyle:
    """Defaults when a stroke's brush isn't in this file's BrushStyle table
    (external brush reference)."""
    return BrushStyle(
        main_id=brush_id,
        pattern_style=0,
        texture_pattern=0,
        hardness=1.0,
        thickness_base=1.0,
        anti_alias=2,
        flow_base=1.0,
        interval_base=1.0,
        auto_interval_type=0,
        rotation_base=0.0,
        rotation_random=0.0,
        rotation_effector=0,
        rotation_in_spray_base=0.0,
        rotation_effector_in_spray=0,
        rotation_random_in_spray=0.0,
        texture_scale=1.0,
        texture_rotate=0.0,
        texture_offset_x=0.0,
        texture_offset_y=0.0,
        texture_density_base=1.0,
        composite_mode=0,
        spray_flag=0,
        spray_size_base=0.0,
        spray_density_base=0.0,
        spray_bias=0.0,
    )


def render_spray_stroke(
    st: VectorStroke,
    brush: BrushStyle,
    buf: np.ndarray,
    ss: int,
    brushes: Dict[int, BrushStyle],
    dfs: Dict[str, pd.DataFrame],
    pattern_images: Dict[int, np.ndarray],
    effector_curves: Dict[int, List[Tuple[float, float]]],
) -> None:
    """Spray brush rendering — scattered pattern stamps along the path.

    The dispatcher routes non-bend brushes through this code path. Walks the
    polyline with an interval-based spacing accumulator
    (``step_px = stamp_size × IntervalBase / 30``); at each step, instead of
    stamping at the curve point, stamps at a random offset within a disk of
    radius derived from ``BrushSpraySize``. Random rotation and per-stamp
    size jitter come from the regular SizeEffector with ``use_random=True``.

    For ``num_stamps > 1`` (multiple stamps per curve point) the scatter
    cluster is replicated; ``SprayDensityBase`` controls that count.

    The ``BrushSprayBias`` 4-branch radius shaping is implemented in
    ``scatter_offset`` below.
    """
    samples = sample_curve_points(st.points, st.vtype, aa_on=(brush.anti_alias != 0))
    if len(samples) < 2:
        return

    color = st.color
    stroke_op = st.stroke_opacity

    canvas_h_ss, canvas_w_ss = buf.shape[:2]

    # Per-stroke alpha plane (Q15). Each stamp accumulates onto this via the
    # Q15 row writer; one final composite happens at stroke end.
    stroke_alpha_q15 = np.zeros((canvas_h_ss, canvas_w_ss), dtype=np.uint16)

    # Secondary canvas-grain texture. ``texture_pattern`` is a foreign-key
    # reference to a pattern image; the grain modulates per-pixel coverage
    # as ``strength = (strength * tex_density_q15) >> 15``. Applied to the
    # stroke alpha plane at composite time.
    secondary_tex: Optional[np.ndarray] = None
    if brush.texture_pattern and brush.texture_pattern in pattern_images:
        secondary_tex = pattern_images[brush.texture_pattern]

    # Spray brushes typically reference multiple pattern bitmaps and cycle
    # among them per stamp for visual variety. The cycling mode lives in
    # ``PatternStyle.OrderType``; ``Reverse2`` selects per-stamp X/Y flips.
    tex_masks: List[np.ndarray] = []
    order_type = 0
    reverse2 = 0
    if brush.pattern_style > 0:
        for iid in get_pattern_style_images(dfs, brush.pattern_style):
            if iid in pattern_images:
                tex_masks.append(pattern_images[iid])
        order_type, reverse2 = get_pattern_style_order(dfs, brush.pattern_style)

    rng = MsvcRandom(seed=st.random_seed or brush.main_id)
    num_stamps = max(1, int(round(brush.spray_density_base)))
    base_rot = np.deg2rad(brush.rotation_base)
    use_random_rot = brush.rotation_random > 0
    # Counter-increment toggle: increments per stamp iff ``StyleFlag`` bit 15
    # is clear. When bit 15 is set, the increment is gated by a pressure-
    # derived value (typically held when pressure is high) — holding the
    # counter for those brushes matches the source renderer better.
    counter_advances = not (brush.style_flag & (1 << 15))
    pattern_counter = 0

    # Cluster radius (max scatter distance) and per-stamp base size.
    # Cluster radius scales with the per-stroke brush size; per-stamp size
    # scales with ``SpraySizeBase``. ``SpraySizeEffector`` random jitter is
    # applied per-stamp inside ``stamp_at_sample``. Multipliers are
    # empirical (best IoU on the reference golden image).
    cluster_radius_base = (float(st.brush_size) / 2.0) * 0.5 * ss
    per_stamp_base_size_canvas = float(brush.spray_size_base) * 0.35

    def emit_stamp(
        cx: float,
        cy: float,
        alpha: float,
        rot: float,
        size: float,
        pattern_idx: int,
        flip_h: bool,
        flip_v: bool,
    ) -> None:
        if cx < 0 or cy < 0 or cx >= canvas_w_ss or cy >= canvas_h_ss:
            return
        flow_q15 = max(0, min(0x8000, int(round(alpha * 0x8000))))
        if tex_masks and pattern_idx >= 0:
            mask = tex_masks[pattern_idx % len(tex_masks)]
            q15_stamp_pattern_into_alpha(
                stroke_alpha_q15,
                mask,
                cx,
                cy,
                size,
                size,
                flow_q15,
                rot,
                flip_h,
                flip_v,
                continuous_plot=False,
            )
        else:
            # Soft disc fallback for brushes without a pattern.
            rx = size / 2.0
            stamp_alpha = np.zeros_like(buf[..., 0])
            disc_alpha_into(
                stamp_alpha,
                cx,
                cy,
                rx,
                rx,
                brush.hardness,
                0.0,
                alpha,
                accumulate=False,
            )
            # Convert to Q15 and additive-blend into stroke buffer.
            disc_q15 = np.minimum(0x8000, (stamp_alpha * 0x8000).astype(np.int32))
            sb = stroke_alpha_q15.astype(np.int32)
            active = sb < disc_q15
            sb = np.where(active, sb + (((disc_q15 - sb) * disc_q15) >> 15), sb)
            np.clip(sb, 0, 0x8000, out=sb)
            stroke_alpha_q15[...] = sb.astype(np.uint16)

    def scatter_offset(cluster_radius: float) -> Tuple[float, float]:
        # Spray scatter: bias <= -1 short-circuits to radius=1 with 1 RNG
        # draw; otherwise a 4-branch shaping law runs on u_a/u_b plus a
        # third draw for theta. Theta is in degrees and pre-truncated to
        # an integer degree to match the source renderer's 1° LUT.
        bias = brush.spray_bias
        if bias <= -1.0:
            radius_factor = 1.0
        else:
            u_a = rng.next_unit()
            u_b = rng.next_unit()
            if bias > 0:
                boundary = 1.0 - bias
                if u_a <= boundary:
                    radius_factor = (boundary * u_b) if (boundary * u_a <= u_b) else u_a
                else:
                    radius_factor = (
                        (1.0 - bias * u_b) if ((1.0 - u_a) / bias <= u_b) else u_a
                    )
            else:  # -1 < bias <= 0
                radius_factor = (bias + 1.0) * max(u_a, u_b) - bias
        u_theta = rng.next_unit()
        radius = radius_factor * cluster_radius
        theta_int = int(u_theta * 360.0) % 360
        s, c = SIN_LUT_DEG[theta_int], COS_LUT_DEG[theta_int]
        return radius * s, radius * c

    def calc_random_factor(eff_random, draw_rng) -> float:
        """Random-modulator factor: ``(1 - min) * r + min`` after one RNG draw."""
        r = draw_rng.next_unit()
        return r * (1.0 - eff_random.min) + eff_random.min

    def stamp_at_sample(s: VectorSample, tangent_rad: float) -> float:
        """Returns the per-stamp diameter so caller can drive ``step_px``.

        RNG draw order: cluster-centre size + density first, then per-stamp
        scatter offset, then per-stamp rotation / size / pattern / opacity.
        """
        nonlocal pattern_counter
        cp = CurvePoint(
            pressure=max(0.1, s.pressure),
            velocity=s.velocity,
            smooth=s.smooth,
            tilt=min(1.0, abs(s.tilt_x)),
        )
        # Cluster-centre parameters.
        # Step 1: size — 1 RNG draw iff ``size_effector.random``.
        size_mult = apply_effector(
            brush.size_effector,
            1.0,
            cp,
            effector_curves,
            rng=rng,
            use_random=True,
        )
        # Step 2: density — 1 RNG draw iff ``spray_density_effector.random``.
        density_factor = apply_effector(
            brush.spray_density_effector,
            1.0,
            cp,
            effector_curves,
            rng=rng,
            use_random=True,
        )
        # Step 3 angle: no RNG draw (bit-test only).
        # The plot-parameter pass is also called per cluster but its
        # effector chains run with random gating disabled for spray, so
        # no extra cluster-centre draws happen there.
        if s.lock_size:
            size_mult = s.size
        cluster_radius = cluster_radius_base * size_mult * s.size
        per_stamp_base = per_stamp_base_size_canvas * ss * size_mult * s.size
        base_alpha = brush.flow_base * stroke_op
        num_local = max(1, int(round(num_stamps * density_factor)))

        for _ in range(num_local):
            # Per-stamp scatter offset within the cluster: 1 RNG draw when
            # ``bias <= -1`` (degenerate radius=1 case), 3 otherwise.
            dx, dy = scatter_offset(cluster_radius)
            cx = s.x * ss + dx
            cy = s.y * ss + dy

            # Per-stamp parameter modulation.
            # Step 1: rotation. Mutually-exclusive add branches on bits
            # 8/6/(9-clear) of ``rotation_effector_in_spray``; then a 1-RNG
            # jitter draw iff bit 7 is set.
            rot_eis = brush.rotation_effector_in_spray
            rot = base_rot + np.deg2rad(brush.rotation_in_spray_base)
            if rot_eis & (1 << 8):
                # bit 8: rot += cluster opacity — not modeled.
                pass
            elif rot_eis & (1 << 6):
                # bit 6: rot += per-control angle field (zero for our records).
                pass
            elif not (rot_eis & (1 << 9)):
                # bit 9 clear → rot += line-direction (stroke tangent).
                rot += tangent_rad
            if rot_eis & (1 << 7):
                rot += brush.rotation_random_in_spray * rng.next_unit() * 2.0 * np.pi
            # Step 2: per-stamp size jitter — 1 RNG draw iff
            # ``spray_size_effector.random`` is present.
            stamp_size = max(1.0, per_stamp_base)
            if (
                brush.spray_size_effector is not None
                and brush.spray_size_effector.random is not None
            ):
                stamp_size *= calc_random_factor(brush.spray_size_effector.random, rng)
            # Step 3: pattern selection. Mode 3 (random) draws 1 RNG;
            # ``reverse2`` X/Y axes each draw 1 if their random branch is
            # selected.
            n_tex = max(1, len(tex_masks))
            if order_type == 3:
                pattern_idx = int(rng.next_unit() * n_tex)
            else:
                pattern_idx = pattern_index_for_counter(
                    pattern_counter, n_tex, order_type
                )
            flag_x = False
            flag_y = False
            if reverse2 & 0x01:
                flag_x = True
            elif reverse2 & 0x02:
                flag_x = rng.next_unit() >= 0.5
            elif reverse2 & 0x04:
                flag_x = bool(pattern_counter & 1)
            if reverse2 & 0x10:
                flag_y = True
            elif reverse2 & 0x20:
                flag_y = rng.next_unit() >= 0.5
            elif reverse2 & 0x40:
                flag_y = bool(pattern_counter & 1)
            flip_h = flag_x ^ flag_y
            flip_v = flag_y
            # Step 4: opacity jitter — 1 RNG draw iff opacity_effector.random.
            stamp_alpha = base_alpha
            if (
                brush.opacity_effector is not None
                and brush.opacity_effector.random is not None
            ):
                stamp_alpha *= calc_random_factor(brush.opacity_effector.random, rng)
            # Step 5: flow jitter — 1 RNG draw iff flow_effector.random.
            if (
                brush.flow_effector is not None
                and brush.flow_effector.random is not None
            ):
                stamp_alpha *= calc_random_factor(brush.flow_effector.random, rng)
            # Step 6: thickness jitter — 1 RNG draw iff thickness_effector.random.
            if (
                brush.thickness_effector is not None
                and brush.thickness_effector.random is not None
            ):
                stamp_size *= calc_random_factor(brush.thickness_effector.random, rng)

            if counter_advances:
                pattern_counter += 1
            emit_stamp(
                cx,
                cy,
                stamp_alpha,
                rot,
                stamp_size,
                pattern_idx,
                flip_h,
                flip_v,
            )
        return per_stamp_base

    # The spacing accumulator threshold is a function of the *current*
    # stamp's diameter, which varies along the curve when the SizeEffector
    # is pressure-driven. Recompute step_px after every stamp.
    accum = 0.0
    last_sample = samples[0]
    init_tan = (
        float(np.arctan2(samples[1].y - samples[0].y, samples[1].x - samples[0].x))
        if len(samples) >= 2
        else 0.0
    )
    last_size = stamp_at_sample(last_sample, init_tan)
    step_px = max(1.0, last_size * brush.interval_base / 30.0)
    for i in range(1, len(samples)):
        s = samples[i]
        dx_ = (s.x - last_sample.x) * ss
        dy_ = (s.y - last_sample.y) * ss
        d = float(np.hypot(dx_, dy_))
        accum += d
        if accum >= step_px:
            tangent = float(np.arctan2(dy_, dx_)) if d > 0 else init_tan
            last_size = stamp_at_sample(s, tangent)
            step_px = max(1.0, last_size * brush.interval_base / 30.0)
            accum = 0.0
        last_sample = s

    # Apply the secondary canvas-grain texture
    # multiplicatively to the stroke alpha plane before compositing. Each
    # buffer pixel `p` is scaled by the wrap-tiled grain at that canvas
    # location: `p = (p * tex_density) >> 15`. `TextureScale` (×ss to map
    # from canvas-pixel space) controls the tiling rate.
    if secondary_tex is not None:
        th, tw = secondary_tex.shape
        scale = max(0.001, brush.texture_scale)
        ys = np.arange(canvas_h_ss)
        xs = np.arange(canvas_w_ss)
        v_idx = (ys / (ss * scale)).astype(np.int32) % th
        u_idx = (xs / (ss * scale)).astype(np.int32) % tw
        grain = secondary_tex[v_idx[:, None], u_idx[None, :]]
        # Brightness/contrast preprocess. Default 0/1 is identity.
        b = brush.texture_brightness if hasattr(brush, "texture_brightness") else 0.0
        c = brush.texture_contrast if hasattr(brush, "texture_contrast") else 1.0
        if b != 0.0 or c != 1.0:
            t = grain.astype(np.float32) / 255.0
            t = np.clip((t - 0.5) * c + 0.5 + b, 0.0, 1.0)
            grain_q15 = (t * 0x8000).astype(np.int32)
        else:
            grain_q15 = (grain.astype(np.int32) * 0x8000) // 255
        sa = stroke_alpha_q15.astype(np.int32)
        sa = (sa * grain_q15) >> 15
        np.clip(sa, 0, 0x8000, out=sa)
        stroke_alpha_q15 = sa.astype(np.uint16)

    # Composite the per-stroke Q15 alpha plane onto the canvas in one
    # alpha-over pass at stroke end (NOT per-stamp blending). This is the
    # key behavioural difference vs the line-stamp path's direct blit.
    composite_q15_alpha_onto(buf, stroke_alpha_q15, color)


def render_vector_line_stamp(
    strokes: List[VectorStroke],
    canvas_shape: Tuple[int, int],
    brush_styles: Optional[pd.DataFrame],
    pattern_images: Dict[int, np.ndarray],
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[Optional[np.ndarray], List[VectorStroke]]:
    """Line-stamp renderer: walk each stroke's sampled path, stamp brush texture.

    Handles ``SprayFlag == 0`` brushes. Returns ``(rendered, skipped)`` where
    ``skipped`` is the list of strokes whose brushes can't be line-stamped
    (currently: any ``SprayFlag != 0`` brush — spray rendering is a future
    addition). The caller is responsible for rasterizing those skipped
    strokes via the legacy Bresenham fallback onto the same canvas.

    ``rendered`` is ``None`` only when the input stroke list is empty after
    skipping; otherwise it's a fully-rendered buffer that may also need
    overlay from skipped strokes.

    Supersampling factor is chosen from the max ``AntiAlias`` level across
    all strokes so the full-canvas buffer can be reused.
    """
    canvas_h, canvas_w = canvas_shape

    # Resolve each stroke's brush row up front.
    brushes: Dict[int, BrushStyle] = {}
    if brush_styles is not None and len(brush_styles) > 0:
        for st in strokes:
            bid = st.brush_id
            if bid in brushes:
                continue
            match = brush_styles[brush_styles["MainId"] == bid]
            brushes[bid] = (
                BrushStyle.from_row(match.iloc[0]) if len(match) else default_brush(bid)
            )

    # Effector curve graphs: {graph_id: control_points}. Effector chains
    # reference these by id; missing ids fall back to identity linear.
    effector_curves: Dict[int, List[Tuple[float, float]]] = {}
    graph_df = dfs.get("BrushEffectorGraphData")
    if graph_df is not None and len(graph_df) > 0:
        for _, row in graph_df.iterrows():
            rec = BrushEffectorGraphDataRecord.from_row(row)
            effector_curves[rec.main_id] = list(rec.control_points)

    ss = max(
        (ANTI_ALIAS_TO_SS.get(b.anti_alias, 2) for b in brushes.values()),
        default=2,
    )
    buf = np.zeros((canvas_h * ss, canvas_w * ss, 4), dtype=np.float32)
    skipped: List[VectorStroke] = []

    for st in strokes:
        color = st.color
        brush_size = st.brush_size
        stroke_op = st.stroke_opacity
        brush_id = st.brush_id
        brush = brushes.get(brush_id) or default_brush(brush_id)

        # Dispatcher: brushes with ``spray_flag != 0`` go through the spray
        # path (scattered pattern stamps); the rest go through the
        # line-stamp path below (analytic disc + MAX-accumulation).
        if brush.spray_flag != 0:
            render_spray_stroke(
                st, brush, buf, ss, brushes, dfs, pattern_images, effector_curves
            )
            continue

        tex_mask: Optional[np.ndarray] = None
        if brush.pattern_style > 0:
            for iid in get_pattern_style_images(dfs, brush.pattern_style):
                if iid in pattern_images:
                    tex_mask = pattern_images[iid]
                    break
        # For disc brushes (no pattern), defer the mask generation to draw
        # time so it's built at the exact stamp size without scaling artifacts.
        use_analytic_disc = tex_mask is None

        samples = sample_curve_points(
            st.points, st.vtype, aa_on=(brush.anti_alias != 0)
        )
        if not samples:
            continue

        # ThicknessBase has two meanings depending on brush type:
        # - For standard pens (AntiAlias > 0): sub-linear scale factor on the
        #   stamp diameter (empirical cube-root fit).
        # - For chisel/brush-tip brushes (AntiAlias == 0): aspect ratio of an
        #   elliptical stamp — long axis stays brush_size, short axis is
        #   brush_size × ThicknessBase.
        elliptical = brush.anti_alias == 0
        if elliptical:
            stamp_long = brush_size * ss
            stamp_short = stamp_long * brush.thickness_base
        else:
            stamp_long = brush_size * float(brush.thickness_base ** (1 / 3)) * ss
            stamp_short = stamp_long
        effective_size = stamp_long  # used for step_px and texture scaling
        # Per-stamp alpha is ``FlowBase``. ``TextureDensityBase`` belongs to
        # the secondary paper-grain overlay, not the primary stamp.
        per_stamp_alpha = brush.flow_base

        # Shaped brushes (pattern_style > 0): texture is stamped as a square,
        # but scaled so the texture's painted stripe (not its transparent
        # padding) matches brush_size. Otherwise the visible line is much
        # thinner than requested. Random rotation is off — nib textures
        # shouldn't spin.
        shaped = brush.pattern_style > 0
        use_random_rot = not shaped
        if shaped:
            tex_h, tex_w = tex_mask.shape
            # For tall textures the painted stripe runs vertically — its width
            # is the count of non-zero columns. For wide textures, it's rows.
            if tex_h >= tex_w:
                col_any = (tex_mask > 0).any(axis=0)
                painted_narrow = max(1, int(col_any.sum()))
                tex_narrow = tex_w
            else:
                row_any = (tex_mask > 0).any(axis=1)
                painted_narrow = max(1, int(row_any.sum()))
                tex_narrow = tex_h
            shaped_scale = float(tex_narrow) / painted_narrow
        else:
            shaped_scale = 1.0

        # Stamp spacing: step = diameter × IntervalBase / 10. Clamped to
        # ≥ 1 px so tiny brushes or a zeroed IntervalBase don't loop forever.
        step_px = max(1.0, effective_size * brush.interval_base / 10.0)

        # Per-stroke seed lifted from the saved blob so re-renders are
        # deterministic against the source's original output. Falls back
        # to the brush id when the stroke didn't store one.
        seed = st.random_seed or brush_id
        rng = np.random.default_rng(seed=seed)
        base_rot = np.deg2rad(brush.rotation_base)

        # Per-stroke alpha accumulator (disc brushes) or full RGBA for textured
        # brushes that still composite each stamp directly.
        stroke_alpha = (
            np.zeros((canvas_h * ss, canvas_w * ss), dtype=np.float32)
            if use_analytic_disc
            else None
        )

        def draw_stamp(s: VectorSample, tangent_rad: float) -> float:
            """Returns the per-stamp size so the caller can adjust step_px."""
            # Per-stamp curve point. Pressure clamped to a 0.1 floor so very-
            # low-pressure samples still register a stamp. The disk only
            # stores tilt_x; the runtime "tilt" axis the effector chain
            # consumes is its magnitude.
            cp = CurvePoint(
                pressure=max(0.1, s.pressure),
                velocity=s.velocity,
                smooth=s.smooth,
                tilt=min(1.0, abs(s.tilt_x)),
            )
            # Size and flow effector chains run with random gating off
            # for the line-stamp path (the random modulator never fires
            # for these channels in this code path).
            size_mult = apply_effector(brush.size_effector, 1.0, cp, effector_curves)
            flow_mult = apply_effector(brush.flow_effector, 1.0, cp, effector_curves)
            # Per-control lock bits override the effector chain when set:
            # bit 12 (lock_size) replaces the size multiplier with the literal
            # cp.size, bit 13 (lock_flow) replaces flow with literal cp.flow.
            if s.lock_size:
                size_mult = s.size
            if s.lock_flow:
                flow_mult = s.flow
            size = (
                effective_size * size_mult * s.size * brush.texture_scale * shaped_scale
            )
            alpha = per_stamp_alpha * flow_mult * stroke_op
            if use_analytic_disc:
                assert stroke_alpha is not None
                # Circular for hard pens, elliptical (rotated by RotationBase)
                # for chisel/brush-tip brushes. ``size`` is the long axis.
                rx = size / 2.0
                ry = (stamp_short / stamp_long) * rx
                # Soft brushes (low hardness) need additive within-stroke
                # accumulation so their falloff tails build up opacity along
                # the stroke. Hard pens use MAX to avoid per-stamp AA boundaries
                # inside the silhouette.
                disc_alpha_into(
                    stroke_alpha,
                    s.x * ss,
                    s.y * ss,
                    rx,
                    ry,
                    brush.hardness,
                    base_rot,
                    alpha,
                    accumulate=brush.hardness < 0.9,
                )
                return size
            rot = base_rot
            if use_random_rot:
                rot += brush.rotation_random * rng.uniform(0, 2 * np.pi)
            elif shaped:
                # Align the texture's bright axis along the stroke tangent.
                # The G-pen texture stores its painted stripe running vertically
                # (along tex-y); rotating by tangent+π/2 turns that stripe into
                # the along-stroke direction for continuous lines.
                rot += tangent_rad + np.pi / 2
            stamp_pattern(
                buf,
                tex_mask,
                s.x * ss,
                s.y * ss,
                size,
                size,
                alpha,
                color,
                rot,
            )
            return size

        accum = 0.0
        last_sample = samples[0]
        if len(samples) >= 2:
            init_tan = float(
                np.arctan2(samples[1].y - samples[0].y, samples[1].x - samples[0].x)
            )
        else:
            init_tan = 0.0
        last_size = draw_stamp(last_sample, init_tan)
        # Stamping spacing is a function of the *current* stamp's
        # diameter, which varies along the curve when the SizeEffector is
        # pressure-driven. Recompute step_px after every stamp.
        step_px = max(1.0, last_size * brush.interval_base / 30.0)
        for i in range(1, len(samples)):
            s = samples[i]
            dx_ = (s.x - last_sample.x) * ss
            dy_ = (s.y - last_sample.y) * ss
            d = float(np.hypot(dx_, dy_))
            accum += d
            if accum >= step_px:
                tangent = float(np.arctan2(dy_, dx_)) if d > 0 else init_tan
                last_size = draw_stamp(s, tangent)
                step_px = max(1.0, last_size * brush.interval_base / 30.0)
                accum = 0.0
            last_sample = s

        # For disc-stamped strokes we accumulated a silhouette via MAX; now
        # alpha-composite it onto the main buffer as a single solid-color pass.
        if use_analytic_disc and stroke_alpha is not None:
            composite_alpha_onto(buf, stroke_alpha, color)

    if ss > 1:
        buf = buf.reshape(canvas_h, ss, canvas_w, ss, 4).mean(axis=(1, 3))
    # CLIP converts coverage→alpha as ``min(255, floor(alpha * 256))``, which
    # matches its internal ``coverage >> 7`` (15-bit coverage down to 8-bit).
    # This differs by one unit from the naïve ``alpha * 255`` at most alpha
    # values.
    rendered = np.minimum(255, (buf * 256).astype(np.int32)).astype(np.uint8)
    return rendered, skipped


def rasterize_vectors(
    clip_data: Dict[str, Union[Dict[int, bytes], bytes, np.ndarray]],
    dfs: Dict[str, pd.DataFrame],
    canvas_shape: Tuple[int, int],
) -> None:
    """Mutate ``clip_data`` in place: each raw-bytes vector entry becomes an
    ``np.ndarray`` render.

    Uses the line-stamp renderer when the brush is in line-stamp mode
    (``SprayFlag == 0``). Falls back to the legacy Bresenham-line renderer for
    spray brushes or when line-stamp cannot resolve a pattern.
    """
    brush_styles = dfs.get("BrushStyle")
    pattern_images = extract_brush_pattern_images(clip_data, dfs)
    for key, value in list(clip_data.items()):
        if not isinstance(value, bytes):
            continue
        strokes = parse_vector_binary(value)
        rendered, skipped = render_vector_line_stamp(
            strokes, canvas_shape, brush_styles, pattern_images, dfs
        )
        if skipped:
            # Spray brushes aren't line-stamped yet; rasterize them as
            # 1-pixel polylines onto the same canvas. Better than dropping
            # the strokes (or punting the entire blob to Bresenham, which
            # discards the line-stamped strokes' anti-aliased thickness).
            poly = rasterize_polylines(skipped, canvas_shape, brush_styles)
            if rendered is None:
                rendered = poly
            else:
                # Source-over composite of `poly` (uint8 RGBA) on top of
                # `rendered` (uint8 RGBA). Both are in 0..255.
                src = poly.astype(np.float32) / 255.0
                dst = rendered.astype(np.float32) / 255.0
                src_a = src[..., 3:4]
                out_a = src_a + dst[..., 3:4] * (1.0 - src_a)
                safe = np.where(out_a > 1e-6, out_a, 1.0)
                out_rgb = (
                    src[..., :3] * src_a + dst[..., :3] * dst[..., 3:4] * (1.0 - src_a)
                ) / safe
                rendered = (np.concatenate([out_rgb, out_a], axis=-1) * 255.0).astype(
                    np.uint8
                )
        if rendered is None:
            rendered = np.zeros((canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8)
        clip_data[key] = rendered
