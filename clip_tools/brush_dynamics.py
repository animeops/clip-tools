"""Per-stamp brush parameter modulation.

Implements the three pieces needed to turn a base brush parameter (e.g.
``BrushSize = 10.0``) into a per-stamp effected value:

1. ``eval_curve_graph(points, x)`` — evaluate a control polyline at
   ``x ∈ [0, 1]``. n=2 → linear; n=3 → quadratic; n>=4 → piecewise cubic
   Bezier with neighbor-midpoint handles.
2. ``apply_effector(eff, base, curve_point, rng_state, curves)`` — chain
   the five modulators in flag-bit order: Pressure → Velocity → Smooth →
   Random → Tilt. Each contributes a multiplier.
3. ``MsvcRandom`` — the standard Microsoft Visual C++ ``rand()`` LCG used
   for per-stamp randomization.

The renderer assembles ``CurvePoint`` instances per polyline sample
(pressure / velocity / smooth / tilt all in [0, 1]) and calls
``apply_effector`` for each base parameter (size, flow, rotation, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from clip_tools.structs.blob_parsers import BrushEffector


# ---------------------------------------------------------------------------
# Per-stamp curve-point: aggregated input axes for the effector chain
# ---------------------------------------------------------------------------


@dataclass
class CurvePoint:
    """Input axes for the effector chain at one polyline sample.

    All fields are [0, 1] by convention (the effector chain assumes this).
    Renderers map raw pen state into this normalized form.
    """

    pressure: float = 1.0
    velocity: float = 0.0
    smooth: float = 0.0
    tilt: float = 0.0


# ---------------------------------------------------------------------------
# Curve graph evaluator (BrushEffectorGraphData.control_points → output)
# ---------------------------------------------------------------------------


def eval_curve_graph(points: List[Tuple[float, float]], x: float) -> float:
    """Evaluate a control polyline as the visual "brush settings → curve" UI.

    For n control points sorted by x:
      - n < 2: returns 0.0
      - n == 2: linear interpolation between the two endpoints (clamped)
      - n == 3: 3-point evaluation via implicit-handle cubic (treats it as
        a 4-point Bezier with the 2 middle points repeated)
      - n >= 4: piecewise cubic. The first segment is linear from p[0]
        to p[1]; the last segment is linear from p[n-2] to p[n-1];
        interior segments use cubic Bezier where each interior control
        point's incoming/outgoing handles are the midpoints of its
        chord with each neighbor.

    Output is the y of the curve at input x. ``x`` outside
    ``[p[0].x, p[-1].x]`` is clamped to the nearest endpoint y, matching
    the brush settings UI.
    """
    n = len(points)
    if n < 2:
        return 0.0

    pts = sorted(points, key=lambda p: p[0])

    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]

    if n == 2:
        (x0, y0), (x1, y1) = pts
        if x1 == x0:
            return y0
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Find segment i such that pts[i].x <= x < pts[i+1].x
    i = 0
    while i + 1 < n and x >= pts[i + 1][0]:
        i += 1
    seg_start = pts[i]
    seg_end = pts[i + 1]

    # Endpoint segments are always linear (no neighbor on the outside).
    if i == 0 or i == n - 2:
        x0, y0 = seg_start
        x1, y1 = seg_end
        if x1 == x0:
            return y0
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Interior segment: cubic Bezier with neighbor-midpoint handles.
    p_prev = pts[i - 1]
    p_next = pts[i + 2]
    handle_in = (
        (seg_start[0] + seg_end[0]) * 0.5,
        (seg_start[1] + seg_end[1]) * 0.5,
    )
    handle_out = (
        (seg_end[0] + p_next[0]) * 0.5,
        (seg_end[1] + p_next[1]) * 0.5,
    )

    # Solve for t such that the cubic Bezier x(t) = x. Newton's method.
    # x(t) = (1-t)^3 P0x + 3(1-t)^2 t H1x + 3(1-t) t^2 H2x + t^3 P1x
    p0x, p0y = seg_start
    p1x, p1y = seg_end
    h1x, h1y = handle_in
    h2x, h2y = handle_out

    # First-order init: linear interp t.
    if p1x == p0x:
        return p0y
    t = (x - p0x) / (p1x - p0x)
    for _ in range(8):
        u = 1.0 - t
        bx = (
            u * u * u * p0x
            + 3 * u * u * t * h1x
            + 3 * u * t * t * h2x
            + t * t * t * p1x
        )
        dbx = (
            3 * u * u * (h1x - p0x) + 6 * u * t * (h2x - h1x) + 3 * t * t * (p1x - h2x)
        )
        if abs(dbx) < 1e-9:
            break
        t -= (bx - x) / dbx
        if t < 0.0:
            t = 0.0
            break
        if t > 1.0:
            t = 1.0
            break
    u = 1.0 - t
    return u * u * u * p0y + 3 * u * u * t * h1y + 3 * u * t * t * h2y + t * t * t * p1y


# ---------------------------------------------------------------------------
# MSVC LCG random — per-stamp RNG used by the renderer
# ---------------------------------------------------------------------------


class MsvcRandom:
    """The Microsoft Visual C++ ``rand()`` linear-congruential generator.

    Used for per-stamp randomization (rotation jitter, Random-flag effector
    modulation, …). State is a 32-bit unsigned integer; each call to
    ``next15`` advances it by ``state = state * 0x41C64E6D + 0x12D687``
    and returns ``(state >> 16) & 0x7FFF`` (a 15-bit value, like
    ``rand()``).
    """

    __slots__ = ("state",)
    MULTIPLIER = 0x41C64E6D
    INCREMENT = 0x12D687
    MASK_32 = 0xFFFFFFFF
    MASK_15 = 0x7FFF

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & self.MASK_32

    def next15(self) -> int:
        """Advance and return a 15-bit pseudorandom integer in [0, 0x7FFF]."""
        self.state = (self.state * self.MULTIPLIER + self.INCREMENT) & self.MASK_32
        return (self.state >> 16) & self.MASK_15

    def next_unit(self) -> float:
        """Pseudorandom float in [0, 1)."""
        return self.next15() / 32768.0


# ---------------------------------------------------------------------------
# Effector chain
# ---------------------------------------------------------------------------


def apply_effector(
    eff: Optional[BrushEffector],
    base: float,
    cp: CurvePoint,
    curves: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    rng: Optional[MsvcRandom] = None,
    use_random: bool = False,
) -> float:
    """Apply the effector chain to a base parameter value.

    Each enabled modulator multiplies the running ``out`` by its factor.
    Disabled (``None``) modulators are skipped. ``curves`` maps a curve id
    (foreign key into ``BrushEffectorGraphData.MainId``) to its decoded
    ``control_points`` list; missing IDs fall back to identity linear
    ``[(0,0), (1,1)]``.

    ``use_random`` gates the Random modulator. The line-stamp size and
    flow paths run with ``use_random=False`` (the random flag bit is
    ignored for those parameters); spray / ribbon paths set it to True.
    The ``rng`` is consumed only when ``use_random`` is True AND the
    Random flag is set.
    """
    if eff is None:
        return base
    out = base

    if eff.pressure is not None:
        cid = eff.pressure.curve_id
        pts = (curves or {}).get(cid, [(0.0, 0.0), (1.0, 1.0)])
        cy = eval_curve_graph(pts, cp.pressure)
        out *= cy * (1.0 - eff.pressure.min) + eff.pressure.min

    if eff.velocity is not None:
        cid = eff.velocity.curve_id
        pts = (curves or {}).get(cid, [(0.0, 0.0), (1.0, 1.0)])
        cy = eval_curve_graph(pts, cp.velocity)
        out *= cy * (eff.velocity.max - eff.velocity.min) + eff.velocity.min

    if eff.smooth is not None:
        out *= (1.0 - cp.smooth) * (1.0 - eff.smooth.min) + eff.smooth.min

    if use_random and eff.random is not None and rng is not None:
        r = rng.next_unit()
        out *= r * (1.0 - eff.random.min) + eff.random.min

    if eff.tilt is not None:
        out *= cp.tilt * (1.0 - eff.tilt.min) + eff.tilt.min

    return out


def build_effector_curves(
    effector_graph_records,
) -> Dict[int, List[Tuple[float, float]]]:
    """Materialize a `{MainId: control_points}` map from a list of
    `BrushEffectorGraphDataRecord`. Pass directly to `apply_effector`.
    """
    return {r.main_id: list(r.control_points) for r in effector_graph_records}
