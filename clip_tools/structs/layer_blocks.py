from typing import List, Tuple
import numpy as np
import pandas as pd
from .offscreen_attributes import process_offscreen_attributes

import struct


def process_layer_blocks(
    blocks: List[Tuple[int, bytes]], offscreen: pd.Series
) -> np.ndarray:
    ALPHA_SUBBLOCK_WIDTH = 64
    PARAMETER_HEADER = "Parameter".encode("utf-16be")
    parameter_header_spec = struct.Struct(">II")

    attr_ds = process_offscreen_attributes(offscreen["Attribute"])

    block_width = attr_ds["block_width"]
    block_height = attr_ds["block_height"]

    num_channels = attr_ds["num_channels"]

    header_index = (
        offscreen["Attribute"].index(PARAMETER_HEADER) + len(PARAMETER_HEADER) + 8
    )
    buff = offscreen["Attribute"][
        header_index : header_index + parameter_header_spec.size
    ]
    num_cols, num_rows = parameter_header_spec.unpack(buff)

    if num_channels == 0:
        buffer = np.zeros(
            (num_rows * block_height, num_cols * block_width), dtype=np.uint8
        )
    elif num_channels == 1:
        buffer = np.zeros(
            (num_rows * block_height, num_cols * block_width, 2), dtype=np.uint8
        )
    else:
        buffer = np.zeros(
            (num_rows * block_height, num_cols * block_width, 4), dtype=np.uint8
        )

    for block_idx, block_data in blocks:
        dt = np.dtype(np.uint8).newbyteorder("<")

        if num_channels == 0:
            shape = [block_height, block_width]
            main_img = np.frombuffer(block_data, dtype=dt).reshape(shape)  # .copy()
        elif num_channels == 1:
            # Brush?
            shape = [block_height * 2, block_width]
            # TODO: Check why this is... only seems to happen for brushes
            # Likely seems to have to do with the fact that CLIP can support 2 brushes
            # For now, only take 1 brush
            temp_img = np.frombuffer(block_data, dtype=dt).reshape(shape)  # .copy()
            main_img = np.zeros((block_height, block_width, 2), dtype=np.uint8)
            main_img[..., 0] = temp_img[0:block_height, :]
            main_img[..., 1] = temp_img[block_height:, :]
            main_img[..., 1] = 255
        else:
            shape = [block_height + ALPHA_SUBBLOCK_WIDTH, block_width, num_channels]
            img_data = np.frombuffer(block_data, dtype=dt).reshape(shape).copy()

            main_img = img_data[ALPHA_SUBBLOCK_WIDTH:]

            mip_0 = img_data[0:ALPHA_SUBBLOCK_WIDTH, 0:ALPHA_SUBBLOCK_WIDTH]
            mip_1 = img_data[
                0:ALPHA_SUBBLOCK_WIDTH, ALPHA_SUBBLOCK_WIDTH : ALPHA_SUBBLOCK_WIDTH * 2
            ]
            mip_2 = img_data[
                0:ALPHA_SUBBLOCK_WIDTH,
                ALPHA_SUBBLOCK_WIDTH * 2 : ALPHA_SUBBLOCK_WIDTH * 3,
            ]
            mip_3 = img_data[
                0:ALPHA_SUBBLOCK_WIDTH,
                ALPHA_SUBBLOCK_WIDTH * 3 : ALPHA_SUBBLOCK_WIDTH * 4,
            ]
            mips = [mip_0, mip_1, mip_2, mip_3]

            stacked = np.concatenate(mips, axis=-1)
            reshaped = stacked.reshape(64, 64, 4, 4)
            swapped = reshaped.swapaxes(1, 2)
            final_array = swapped.reshape(block_height, block_width)
            main_img[..., 3] = final_array

            # Create the image
            main_img[:, :, [0, 2]] = main_img[:, :, [2, 0]]

        buffer[
            (block_idx // num_cols) * block_height : ((block_idx // num_cols) + 1)
            * block_height,
            (block_idx % num_cols) * block_width : ((block_idx % num_cols) + 1)
            * block_width,
        ] = main_img

    return buffer[: attr_ds["height"], : attr_ds["width"]]
