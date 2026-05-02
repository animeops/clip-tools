"""Per-channel and HSY blend mode formulas for `Layer.LayerComposite`.

Argument convention: every function takes `(base, blend)` where `base` is
the bottom layer and `blend` is the top layer. Per-channel functions
operate elementwise on uint8 arrays (any shape; typically `(H, W)` per
channel or `(H, W, 3)` for RGB). HSY functions take RGB triples
`(..., 3)` and return RGB triples.

The `+ 127` then `// 255` pattern is CSP's standard round-to-nearest div.
"""

from typing import Callable

import numpy as np

from clip_tools.constants import LayerComposite


# ---------------------------------------------------------------------------
# Per-channel formulas
# ---------------------------------------------------------------------------


def _i32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int32)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)


def normal(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return blend.copy()


def multiply(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8((_i32(base) * _i32(blend) + 127) // 255)


def screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    return _u8(255 - ((255 - b) * (255 - s) + 127) // 255)


def overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    low = (2 * b * s + 127) // 255
    t = 2 * b - 255
    high = ((t + s) * 255 - t * s + 127) // 255
    return _u8(np.where(b < 128, low, high))


def hard_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    low = (2 * s * b + 127) // 255
    t = 2 * s - 255
    high = ((t + b) * 255 - t * b + 127) // 255
    return _u8(np.where(s < 128, low, high))


def darken(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.minimum(base, blend)


def lighten(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.maximum(base, blend)


def color_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    inv_s = 255 - s
    safe_inv_s = np.where(inv_s == 0, 1, inv_s)
    quotient = np.minimum(255, (b * 255 + inv_s // 2) // safe_inv_s)
    out = np.where(b == 0, 0, np.where(s == 255, 255, quotient))
    return _u8(out)


def color_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    safe_s = np.where(s == 0, 1, s)
    # (S + B) * 255 + S/2 - 255*255 over S, max(0, ...)
    quotient = np.maximum(0, ((s + b) * 255 + s // 2 - 65025) // safe_s)
    out = np.where(s == 255, 255, np.where(s == 0, 0, quotient))
    return _u8(out)


def linear_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8(_i32(base) + _i32(blend) - 255)


def linear_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8(_i32(base) + _i32(blend))


def subtract(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8(_i32(base) - _i32(blend))


def divide(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    safe_s = np.where(s == 0, 1, s)
    quotient = np.minimum(255, (b * 255 + s // 2) // safe_s)
    out = np.where(b == 0, 0, np.where(s == 0, 255, quotient))
    return _u8(out)


def difference(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8(np.abs(_i32(base) - _i32(blend)))


def exclusion(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    return _u8(((b + s) * 255 - 2 * b * s + 127) // 255)


def soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base).astype(np.int64)
    s = _i32(blend).astype(np.int64)

    # f(B) piecewise:
    #   if S < 128:        f = (255 - B) * B
    #   elif B < 64:       f = ((16*B - 3060) * B + 195075) * B / 255
    #   else:              f = sqrt(B*255) * 255 - B * 255 + 0.5
    f_low = (255 - b) * b
    f_mid = (((16 * b - 3060) * b + 195075) * b + 127) // 255
    f_high = (np.sqrt(b * 255).astype(np.float64) * 255.0 + 0.5).astype(
        np.int64
    ) - b * 255
    f = np.where(s < 128, f_low, np.where(b < 64, f_mid, f_high))

    out = (f * (2 * s - 255) + b * 65025 + 32512) // 65025
    return _u8(out)


def linear_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return _u8(_i32(base) + 2 * _i32(blend) - 255)


def pin_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    low = np.minimum(b, 2 * s)
    high = np.maximum(b, 2 * s - 255)
    return _u8(np.where(s < 128, low, high))


def vivid_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    inv_b = 255 - b

    # Color Burn branch (S < 128): R = max(0, 255 - (255-B)*255 / (2*S))
    twice_s = np.maximum(2 * s, 1)
    burn_branch = np.maximum(0, 255 - (inv_b * 255 + s) // twice_s)

    # Color Dodge branch (S >= 128): R = clamp_255(B*255 / (2*(255-S)))
    twice_inv_s = np.maximum(510 - 2 * s, 1)
    dodge_branch = np.minimum(255, (b * 255 + (510 - 2 * s) // 2) // twice_inv_s)

    out = np.where(
        s == 0,
        0,
        np.where(
            s == 255,
            255,
            np.where(
                s < 128,
                np.where(b == 255, 255, burn_branch),
                np.where(b == 0, 0, dodge_branch),
            ),
        ),
    )
    return _u8(out)


def hard_mix(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    b = _i32(base)
    s = _i32(blend)
    bias = np.where(b < 128, 1, 0)
    return _u8(np.where(b + s - bias > 254, 255, 0))


# ---------------------------------------------------------------------------
# HSY-based formulas (operate on RGB triples)
# ---------------------------------------------------------------------------

# CSP uses Rec 601 luma as the Y in HSY. The 16.16 fixed-point form
# `Y = (R*0x4ccd + G*0x970a + B*0x1c29 + 0x8000) >> 16` is mathematically
# (≈) `0.299 R + 0.589 G + 0.110 B`.
LUMA_R = 19661
LUMA_G = 38666
LUMA_B = 7209


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec-601 luma using CSP's 16.16 fixed-point coefficients. Returns
    `int32` in [0, 255]."""
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    y = (r * LUMA_R + g * LUMA_G + b * LUMA_B + 32768) >> 16
    return y.astype(np.int32)


def _clip_color(rgb: np.ndarray) -> np.ndarray:
    """W3C `ClipColor`: scale toward the current luma so all channels land
    in [0, 255]. Input is a float `(..., 3)` array; output is the same."""
    L = (
        rgb[..., 0] * (LUMA_R / 65536.0)
        + rgb[..., 1] * (LUMA_G / 65536.0)
        + rgb[..., 2] * (LUMA_B / 65536.0)
    )
    n = rgb.min(axis=-1)
    x = rgb.max(axis=-1)

    out = rgb.copy()
    # If min < 0: out = L + (out - L) * L / (L - n)
    mask = n < 0
    if np.any(mask):
        denom = np.where(mask, L - n, 1.0)
        scale = np.where(mask, L / denom, 1.0)
        out = np.where(
            mask[..., None],
            L[..., None] + (rgb - L[..., None]) * scale[..., None],
            out,
        )
    # Recompute extremes against the (possibly-updated) array.
    x = out.max(axis=-1)
    # If max > 255: out = L + (out - L) * (255 - L) / (x - L)
    mask = x > 255
    if np.any(mask):
        denom = np.where(mask, x - L, 1.0)
        scale = np.where(mask, (255 - L) / denom, 1.0)
        out = np.where(
            mask[..., None],
            L[..., None] + (out - L[..., None]) * scale[..., None],
            out,
        )
    return out


def _set_luma(rgb: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    """Replace the luma of `rgb` with `new_y` while preserving hue + sat.
    `rgb` is `(..., 3)` (any int/float); `new_y` is `(...)`. Returns uint8
    `(..., 3)`."""
    rgbf = rgb.astype(np.float64)
    yf = new_y.astype(np.float64)
    cur = (
        rgbf[..., 0] * (LUMA_R / 65536.0)
        + rgbf[..., 1] * (LUMA_G / 65536.0)
        + rgbf[..., 2] * (LUMA_B / 65536.0)
    )
    out = rgbf + (yf - cur)[..., None]
    out = _clip_color(out)
    return _u8(np.round(out))


def _set_sat(rgb: np.ndarray, new_sat: np.ndarray) -> np.ndarray:
    """Replace the saturation (defined as max-min) of `rgb` with `new_sat`
    while preserving hue (channel ordering) and centroid. W3C `SetSat`."""
    rgbf = rgb.astype(np.float64)
    mn = rgbf.min(axis=-1)
    mx = rgbf.max(axis=-1)
    cur_sat = mx - mn
    target = new_sat.astype(np.float64)
    out = np.zeros_like(rgbf)
    has_sat = cur_sat > 0
    if np.any(has_sat):
        # Per-pixel: out[c] = (rgb[c] - mn) * target / cur_sat
        scale = np.where(has_sat, target / np.where(cur_sat == 0, 1.0, cur_sat), 0.0)
        out = (rgbf - mn[..., None]) * scale[..., None]
    return out


def hue_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Blend's hue, base's sat, base's luma."""
    bf = base.astype(np.float64)
    sf = blend.astype(np.float64)
    base_sat = bf.max(axis=-1) - bf.min(axis=-1)
    base_luma = _luma(base).astype(np.float64)
    sat_replaced = _set_sat(sf, base_sat)
    return _set_luma(sat_replaced, base_luma)


def saturation_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Base's hue, blend's sat, base's luma."""
    sf = blend.astype(np.float64)
    bf = base.astype(np.float64)
    blend_sat = sf.max(axis=-1) - sf.min(axis=-1)
    base_luma = _luma(base).astype(np.float64)
    sat_replaced = _set_sat(bf, blend_sat)
    return _set_luma(sat_replaced, base_luma)


def color_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Blend's hue + sat, base's luma."""
    return _set_luma(blend, _luma(base).astype(np.float64))


def luminosity_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Base's hue + sat, blend's luma."""
    return _set_luma(base, _luma(blend).astype(np.float64))


def darker_color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Pick the whole pixel from whichever side has lower luma."""
    yb = _luma(base)
    ys = _luma(blend)
    take_blend = ys < yb
    return np.where(take_blend[..., None], blend, base)


def lighter_color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    yb = _luma(base)
    ys = _luma(blend)
    take_blend = ys > yb
    return np.where(take_blend[..., None], blend, base)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Modes whose RGB result is per-channel (apply the function elementwise to
# each of R, G, B independently).
_PER_CHANNEL: dict = {
    LayerComposite.NORMAL: normal,
    LayerComposite.DARKEN: darken,
    LayerComposite.MULTIPLY: multiply,
    LayerComposite.COLOR_BURN: color_burn,
    LayerComposite.LINEAR_BURN: linear_burn,
    LayerComposite.SUBTRACT: subtract,
    LayerComposite.LIGHTEN: lighten,
    LayerComposite.SCREEN: screen,
    LayerComposite.COLOR_DODGE: color_dodge,
    # GLOW_DODGE folds to COLOR_DODGE per-channel; alpha treatment differs
    # (block-render variant) and is not yet wired in.
    LayerComposite.GLOW_DODGE: color_dodge,
    LayerComposite.ADD: linear_dodge,
    LayerComposite.ADD_GLOW: linear_dodge,
    LayerComposite.OVERLAY: overlay,
    LayerComposite.SOFT_LIGHT: soft_light,
    LayerComposite.HARD_LIGHT: hard_light,
    LayerComposite.VIVID_LIGHT: vivid_light,
    LayerComposite.LINEAR_LIGHT: linear_light,
    LayerComposite.PIN_LIGHT: pin_light,
    LayerComposite.HARD_MIX: hard_mix,
    LayerComposite.DIFFERENCE: difference,
    LayerComposite.EXCLUSION: exclusion,
    LayerComposite.DIVIDE: divide,
}

# Modes whose RGB result needs the full triplet at once.
_HSY: dict = {
    LayerComposite.DARKER_COLOR: darker_color,
    LayerComposite.LIGHTER_COLOR: lighter_color,
    LayerComposite.HUE: hue_blend,
    LayerComposite.SATURATION: saturation_blend,
    LayerComposite.COLOR: color_blend,
    LayerComposite.LUMINOSITY: luminosity_blend,
}


def blend_rgb(
    base_rgb: np.ndarray, blend_rgb: np.ndarray, mode: LayerComposite
) -> np.ndarray:
    """Apply `mode` to two RGB arrays of shape `(..., 3)`. Returns uint8.

    Modes that don't have a per-channel formula (the four HSY modes plus
    Darker/Lighter Color) operate on the whole triplet.

    `PASS_THROUGH` and unknown modes fall back to NORMAL.
    """
    fn = _PER_CHANNEL.get(mode)
    if fn is not None:
        return fn(base_rgb, blend_rgb)
    fn = _HSY.get(mode)
    if fn is not None:
        return fn(base_rgb, blend_rgb)
    # PASS_THROUGH on a non-folder layer, or any unknown int — treat as Normal.
    return normal(base_rgb, blend_rgb)


def composite_layer(
    base_rgba: np.ndarray, blend_rgba: np.ndarray, mode: LayerComposite
) -> np.ndarray:
    """Composite `blend_rgba` onto `base_rgba` using `mode` for the RGB
    channels and standard alpha-over for the alpha channel.

    Both inputs must be uint8 RGBA `(H, W, 4)` in straight (non-premultiplied)
    alpha. Output is the same shape.

    Math (W3C compositing-and-blending, simple-alpha-compositing variant)::

        Cs_blended = blend(Cb, Cs)              # per `mode`
        Co = (1 - αs) * Cb + αs * Cs_blended    # straight alpha over
        αo = αs + αb * (1 - αs)
    """
    cb_rgb = base_rgba[..., :3]
    cs_rgb = blend_rgba[..., :3]
    cb_a = base_rgba[..., 3:4].astype(np.float32) / 255.0
    cs_a = blend_rgba[..., 3:4].astype(np.float32) / 255.0

    cs_blended = blend_rgb(cb_rgb, cs_rgb, mode).astype(np.float32)
    cb = cb_rgb.astype(np.float32)

    out_rgb = (1.0 - cs_a) * cb + cs_a * cs_blended
    out_a = cs_a + cb_a * (1.0 - cs_a)

    out_rgb = np.clip(out_rgb, 0, 255)
    out_a = np.clip(out_a * 255.0, 0, 255)
    return np.concatenate([out_rgb, out_a], axis=-1).round().astype(np.uint8)


__all__ = [
    "blend_rgb",
    "composite_layer",
    "color_blend",
    "color_burn",
    "color_dodge",
    "darken",
    "darker_color",
    "difference",
    "divide",
    "exclusion",
    "hard_light",
    "hard_mix",
    "hue_blend",
    "lighten",
    "lighter_color",
    "linear_burn",
    "linear_dodge",
    "linear_light",
    "luminosity_blend",
    "multiply",
    "normal",
    "overlay",
    "pin_light",
    "saturation_blend",
    "screen",
    "soft_light",
    "subtract",
    "vivid_light",
]
