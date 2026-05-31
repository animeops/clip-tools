"""Encode raster pixel data back into the per-block layout used by Offscreen
BlockData chunks.

This is the inverse of :func:`process_layer_blocks`. A layer's raster is stored
as a grid of fixed-size blocks (typically 256x256). For a 4-channel (RGBA) layer
each block's decompressed payload is laid out as ``(block_height + 64, block_width,
4)`` bytes:

- Rows ``[64:]`` hold the color section. Channels are stored ``B, G, R`` with a
  4th channel that the decoder overwrites and ignores (stale buffer data in
  files written by the editor; see ``meaningful`` note below).
- Rows ``[0:64]`` hold the alpha plane, tiled: the 256x256 alpha is folded into a
  64x64 grid of 4x4 super-pixels and split across four 64-wide sub-tiles.

Only the color bytes and the tiled alpha plane carry meaning. Round-tripping
arbitrary RGBA through :func:`encode_pixel_block` -> :func:`decode_pixel_block`
is exact; round-tripping an editor-written block back to bytes reproduces every
meaningful byte (the ignored 4th color channel is normalized to the alpha value).
"""

from typing import Dict, Tuple

import numpy as np

ALPHA_SUBBLOCK_WIDTH = 64


def decode_pixel_block(
    block_data: bytes, block_width: int, block_height: int
) -> np.ndarray:
    """Decode one RGBA block payload into an ``(block_height, block_width, 4)`` array."""
    img = (
        np.frombuffer(block_data, dtype=np.uint8)
        .reshape(block_height + ALPHA_SUBBLOCK_WIDTH, block_width, 4)
        .copy()
    )

    color = img[ALPHA_SUBBLOCK_WIDTH:].copy()

    mips = [
        img[
            0:ALPHA_SUBBLOCK_WIDTH,
            ALPHA_SUBBLOCK_WIDTH * k : ALPHA_SUBBLOCK_WIDTH * (k + 1),
        ]
        for k in range(4)
    ]
    alpha = (
        np.concatenate(mips, axis=-1)
        .reshape(ALPHA_SUBBLOCK_WIDTH, ALPHA_SUBBLOCK_WIDTH, 4, 4)
        .swapaxes(1, 2)
        .reshape(block_height, block_width)
    )

    color[..., 3] = alpha
    color[:, :, [0, 2]] = color[:, :, [2, 0]]  # B,G,R -> R,G,B
    return color


def encode_pixel_block(block_rgba: np.ndarray) -> bytes:
    """Encode an ``(block_height, block_width, 4)`` RGBA array into a block payload.

    Inverse of :func:`decode_pixel_block`. The block must be 256x256 -- the tiled
    alpha layout folds it into a 64x64 grid of 4x4 super-pixels.
    """
    if block_rgba.ndim != 3 or block_rgba.shape[2] != 4:
        raise ValueError("block_rgba must be (h, w, 4) RGBA")
    h, w = block_rgba.shape[:2]
    side = ALPHA_SUBBLOCK_WIDTH * 4  # the tiled alpha layout fixes blocks to 256x256
    if (h, w) != (side, side):
        raise ValueError(f"block must be {side}x{side}, got {h}x{w}")

    block_rgba = np.ascontiguousarray(block_rgba, dtype=np.uint8)
    out = np.zeros((h + ALPHA_SUBBLOCK_WIDTH, w, 4), dtype=np.uint8)

    color = out[ALPHA_SUBBLOCK_WIDTH:]
    color[..., 0] = block_rgba[..., 2]  # B
    color[..., 1] = block_rgba[..., 1]  # G
    color[..., 2] = block_rgba[..., 0]  # R
    color[..., 3] = block_rgba[..., 3]  # normalized to alpha (decoder ignores this)

    alpha = block_rgba[..., 3]
    stacked = (
        alpha.reshape(ALPHA_SUBBLOCK_WIDTH, 4, ALPHA_SUBBLOCK_WIDTH, 4)
        .swapaxes(1, 2)
        .reshape(ALPHA_SUBBLOCK_WIDTH, ALPHA_SUBBLOCK_WIDTH, 16)
    )
    for k in range(4):
        out[
            0:ALPHA_SUBBLOCK_WIDTH,
            ALPHA_SUBBLOCK_WIDTH * k : ALPHA_SUBBLOCK_WIDTH * (k + 1),
        ] = stacked[:, :, 4 * k : 4 * k + 4]
    return out.tobytes()


def tile_image_to_blocks(
    image_rgba: np.ndarray,
    block_width: int = 256,
    block_height: int = 256,
    skip_empty: bool = True,
) -> Tuple[Dict[int, bytes], int, int]:
    """Tile a full-canvas RGBA image into encoded block payloads.

    Returns ``(blocks, num_cols, num_rows)`` where ``blocks`` maps a row-major
    block index to its encoded payload. Blocks whose alpha is entirely zero are
    omitted when ``skip_empty`` is set (matching how the editor stores sparse
    rasters); the canvas edge is zero-padded to a whole number of blocks.
    """
    if image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
        raise ValueError("image_rgba must be (h, w, 4) RGBA")
    height, width = image_rgba.shape[:2]
    num_cols = (width + block_width - 1) // block_width
    num_rows = (height + block_height - 1) // block_height

    blocks: Dict[int, bytes] = {}
    for row in range(num_rows):
        for col in range(num_cols):
            tile = np.zeros((block_height, block_width, 4), dtype=np.uint8)
            y0, x0 = row * block_height, col * block_width
            chunk = image_rgba[y0 : y0 + block_height, x0 : x0 + block_width]
            tile[: chunk.shape[0], : chunk.shape[1]] = chunk
            if skip_empty and not tile[..., 3].any():
                continue
            blocks[row * num_cols + col] = encode_pixel_block(tile)
    return blocks, num_cols, num_rows
