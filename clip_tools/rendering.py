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

from clip_tools.constants import VectorType
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
      come out smooth. See unknowns.md.
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
            w = p0.width_factor + t * (p1.width_factor - p0.width_factor)
            op = p0.opacity_factor + t * (p1.opacity_factor - p0.opacity_factor)
            smod = p0.size_modulation + t * (p1.size_modulation - p0.size_modulation)
            out.append(
                VectorSample(
                    x=x + dx,
                    y=y + dy,
                    pressure=press,
                    width_factor=w,
                    opacity_factor=op,
                    size_modulation=smod,
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


def render_vector_line_stamp(
    strokes: List[VectorStroke],
    canvas_shape: Tuple[int, int],
    brush_styles: Optional[pd.DataFrame],
    pattern_images: Dict[int, np.ndarray],
    dfs: Dict[str, pd.DataFrame],
) -> Optional[np.ndarray]:
    """Line-stamp renderer: walk each stroke's sampled path, stamp brush texture.

    Handles ``SprayFlag == 0`` brushes. Returns ``None`` if any stroke uses a
    spray brush (caller should fall back to the legacy Bresenham renderer —
    the spray algorithm is a future addition).

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

    ss = max(
        (ANTI_ALIAS_TO_SS.get(b.anti_alias, 2) for b in brushes.values()),
        default=2,
    )
    buf = np.zeros((canvas_h * ss, canvas_w * ss, 4), dtype=np.float32)

    for st in strokes:
        color = st.color
        brush_size = st.brush_size
        stroke_op = st.stroke_opacity
        brush_id = st.brush_id
        brush = brushes.get(brush_id) or default_brush(brush_id)

        if brush.spray_flag != 0:
            return None  # caller falls back to Bresenham

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
        # Per-stamp alpha: FlowBase. Density only dims textured brushes.
        per_stamp_alpha = brush.flow_base
        if brush.pattern_style > 0:
            per_stamp_alpha *= brush.texture_density_base

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

        rng = np.random.default_rng(seed=brush_id)
        base_rot = np.deg2rad(brush.rotation_base)

        # Per-stroke alpha accumulator (disc brushes) or full RGBA for textured
        # brushes that still composite each stamp directly.
        stroke_alpha = (
            np.zeros((canvas_h * ss, canvas_w * ss), dtype=np.float32)
            if use_analytic_disc
            else None
        )

        def draw_stamp(s: VectorSample, tangent_rad: float) -> None:
            press = max(0.1, s.pressure)
            size = (
                effective_size
                * s.width_factor
                * press
                * brush.texture_scale
                * shaped_scale
            )
            alpha = per_stamp_alpha * stroke_op * s.opacity_factor * press
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
                return
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

        accum = 0.0
        last_sample = samples[0]
        if len(samples) >= 2:
            init_tan = float(
                np.arctan2(samples[1].y - samples[0].y, samples[1].x - samples[0].x)
            )
        else:
            init_tan = 0.0
        draw_stamp(last_sample, init_tan)
        for i in range(1, len(samples)):
            s = samples[i]
            dx_ = (s.x - last_sample.x) * ss
            dy_ = (s.y - last_sample.y) * ss
            d = float(np.hypot(dx_, dy_))
            accum += d
            if accum >= step_px:
                tangent = float(np.arctan2(dy_, dx_)) if d > 0 else init_tan
                draw_stamp(s, tangent)
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
    return np.minimum(255, (buf * 256).astype(np.int32)).astype(np.uint8)


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
        rendered = render_vector_line_stamp(
            strokes, canvas_shape, brush_styles, pattern_images, dfs
        )
        if rendered is None:
            rendered = rasterize_polylines(strokes, canvas_shape, brush_styles)
        clip_data[key] = rendered
