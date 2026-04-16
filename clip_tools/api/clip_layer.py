from __future__ import annotations

import os
from typing import List, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
from PIL import Image

import logging

from clip_tools.constants import DEBUG
from clip_tools.structs import (
    process_text_attributes,
    process_resizable_image_attributes,
)
from clip_tools.utils import (
    calculate_homography,
    backward_mapping,
    alpha_blend,
    arr_to_pil,
)

logger = logging.getLogger(__name__)


class ClipLayer:
    text_attributes = {}
    resizable_image_attributes = {}
    composite_cache = {}

    def __init__(
        self,
        record: pd.DataFrame,
        idx: int,
        raster: dict,
        canvas_size: Tuple[int, int],
        layer_map: Optional[dict] = None,
    ):
        """
        Args:
            record (pd.DataFrame): Layer dataframe.
            idx (int): Index of the layer in the layer dataframe.
            raster (dict): Map from layer_id to the raster image.
            canvas_size (Tuple[int, int]): Size of the canvas.
            layer_map (Optional[dict]): Map from layer_id to index in the layer dataframe.
        """
        self._record = record
        self._idx = idx
        self._raster = raster
        self._canvas_size = canvas_size

        if layer_map is None:
            self._layer_map = {}
            for index, row in self._record.iterrows():
                self._layer_map[row["MainId"]] = index
        else:
            self._layer_map = layer_map

        self._children = []
        for child_id in self.children_ids:
            self._children.append(
                ClipLayer(
                    self._record,
                    self._layer_map[child_id],
                    self._raster,
                    self._canvas_size,
                    self._layer_map,
                )
            )

    @property
    def layer_id(self) -> int:
        return self._record["MainId"].loc[self._idx]

    @property
    def name(self) -> str:
        return self._record["LayerName"].loc[self._idx]

    @property
    def layer_type(self) -> str:
        if self.layer_id in self._raster:
            return self._raster[self.layer_id]["type"]
        else:
            return "raster"

    @property
    def parent_id(self) -> int:
        return self._record["ParentLayer"].loc[self._idx]

    @property
    def children_ids(self) -> List[int]:
        return list(
            self._record[self._record["ParentLayer"] == self.layer_id]["MainId"].values
        )

    @property
    def children_names(self) -> List[str]:
        return [child.name for child in self._children]

    @property
    def parent(self) -> ClipLayer:
        return ClipLayer(
            self._record,
            self._layer_map[self.parent_id],
            self._raster,
            self._canvas_size,
            self._layer_map,
        )

    @property
    def children(self) -> List[ClipLayer]:
        return self._children

    @property
    def visible(self) -> bool:
        return self._record["LayerVisibility"].loc[self._idx] != 0

    @property
    def opacity(self) -> float:
        ## TODO: WILL LIKELY FAIL
        return float(self._record["LayerOpacity"].loc[self._idx]) / 256.0

    @opacity.setter
    def opacity(self, value):
        ## TODO: WILL LIKELY FAIL
        self._record.loc[self._idx, "LayerOpacity"] = int(value * 256)

    @visible.setter
    def visible(self, value):
        self._record.loc[self._idx, "LayerVisibility"] = 1 if value else 0

    @property
    def size(self) -> Tuple[int, int]:
        """Will return width, height."""
        return self._canvas_size[::-1][0], self._canvas_size[::-1][1]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        layer_metadata = self._record.loc[self._idx]

        if "TextLayerType" in layer_metadata.keys() and not np.isnan(
            layer_metadata["TextLayerType"]
        ):
            if DEBUG:
                if not os.path.exists("temp/text_attributes"):
                    os.makedirs("temp/text_attributes")
                with open(
                    f"temp/text_attributes/{self.layer_id}_text_attribute.binary",
                    mode="wb",
                ) as f:
                    f.write(layer_metadata["TextLayerAttributes"])

            try:
                if self.layer_id not in ClipLayer.text_attributes:
                    ClipLayer.text_attributes[self.layer_id] = process_text_attributes(
                        layer_metadata["TextLayerAttributes"]
                    )

                attr_ds = ClipLayer.text_attributes[self.layer_id]

                layer_offset_x = (
                    layer_metadata["LayerOffsetX"] + attr_ds["general_offset_x"]
                )
                layer_offset_y = (
                    layer_metadata["LayerOffsetY"] + attr_ds["general_offset_y"]
                )
            except Exception:
                print(
                    f"WARNING: Text attribute layer load failed for layer_id: {self.layer_id}"
                )
                layer_offset_x = layer_metadata["LayerOffsetX"]
                layer_offset_y = layer_metadata["LayerOffsetY"]

        else:
            layer_offset_x = 0
            layer_offset_y = 0

        if "DrawToRenderOffscreenType" in layer_metadata.keys() and not np.isnan(
            layer_metadata["DrawToRenderOffscreenType"]
        ):
            pass
        else:
            if (
                "LayerRenderOffscrOffsetX" in layer_metadata.keys()
                and layer_metadata["LayerRenderOffscrOffsetX"]
            ):
                layer_offset_x += layer_metadata["LayerRenderOffscrOffsetX"]
            if (
                "LayerRenderOffscrOffsetY" in layer_metadata.keys()
                and layer_metadata["LayerRenderOffscrOffsetY"]
            ):
                layer_offset_y += layer_metadata["LayerRenderOffscrOffsetY"]

        if "LayerOffsetX" in layer_metadata.keys() and layer_metadata["LayerOffsetX"]:
            layer_offset_x += layer_metadata["LayerOffsetX"]
        if "LayerOffsetY" in layer_metadata.keys() and layer_metadata["LayerOffsetY"]:
            layer_offset_y += layer_metadata["LayerOffsetY"]

        return (layer_offset_x, layer_offset_y, 0, 0)

    def is_group(self) -> bool:
        return self._record.loc[self._idx]["LayerFolder"] in [1, 17]

    def __iter__(self) -> Iterator[ClipLayer]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def composite(self, prepended_layers=[]) -> Optional[Image.Image]:
        if self.layer_id in self._raster:
            composited = self._raster[self.layer_id]["image"]
        elif self.layer_id in ClipLayer.composite_cache:
            composited = ClipLayer.composite_cache[self.layer_id]
        elif (
            "DrawColorEnable" in self._record.keys()
            and self._record["DrawColorEnable"].loc[self._idx] == 1.0
        ):
            red = 255 * (self._record["DrawColorMainRed"].loc[self._idx] / (2**32 - 1))
            green = 255 * (
                self._record["DrawColorMainGreen"].loc[self._idx] / (2**32 - 1)
            )
            blue = 255 * (
                self._record["DrawColorMainBlue"].loc[self._idx] / (2**32 - 1)
            )
            bg_color = np.array([red, green, blue, 255], dtype=np.uint8)
            composited = np.zeros((*self._canvas_size, 4), dtype=np.uint8)
            composited[..., :4] = bg_color
        else:
            logger.debug("Composit children:", self.children_names)

            if len(self.children) > 0:
                layers_to_composit = prepended_layers + self.children
                composited = self._composit_layers(layers_to_composit)
            else:
                return None

        ClipLayer.composite_cache[self.layer_id] = composited
        return arr_to_pil(composited)

    def _composit_layers(self, layer_list: List[ClipLayer]) -> np.ndarray:
        """
        layer_list is a list of tuples (id, image)
        """

        buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

        buffer[..., :3] = 255

        for i, layer in enumerate(layer_list):
            layer_id = layer.layer_id
            layer_type = layer.layer_type
            layer_img = np.array(layer.composite())

            if len(layer_img.shape) == 0:
                continue

            layer_metadata = self._record.loc[self._layer_map[layer_id]]

            if (
                "ResizableImageInfo" in layer_metadata.keys()
                and layer_metadata["ResizableImageInfo"]
            ):
                if DEBUG:
                    # Save as binary
                    if not os.path.exists("temp/resizable_image_info"):
                        os.makedirs("temp/resizable_image_info")
                    with open(
                        f"temp/resizable_image_info/{layer_id}_resizable_image_info.binary",
                        mode="wb",
                    ) as f:
                        f.write(layer_metadata["ResizableImageInfo"])

                ClipLayer.resizable_image_attributes[layer_id] = (
                    process_resizable_image_attributes(
                        layer_metadata["ResizableImageInfo"]
                    )
                )

                layer_buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

                source_coords = ClipLayer.resizable_image_attributes[layer_id][
                    "source_coords"
                ]
                polygon_coords = ClipLayer.resizable_image_attributes[layer_id][
                    "polygon_coords"
                ]
                transform = calculate_homography(source_coords, polygon_coords)
                layer_buffer = backward_mapping(
                    transform, layer_img, layer_buffer, polygon_coords
                )
                buffer = alpha_blend(buffer, layer_buffer, premultiplied=False)

            else:
                if layer_img is None:
                    logger.debug(f"Skipping None layer idx: {i}, id: {layer_id}")
                    continue

                if layer_metadata["LayerVisibility"] == 0:
                    logger.debug(f"Skipping {layer.name}")
                    continue

                layer_offset_x, layer_offset_y = layer.bbox[:2]
                layer_height, layer_width = layer_img.shape[:2]

                if layer_offset_x < 0:
                    layer_width = layer_img.shape[1] + layer_offset_x
                    layer_offset_x = 0
                    layer_img = layer_img[:, -layer_width:]
                else:
                    if layer_width + layer_offset_x > buffer.shape[1]:
                        layer_width = buffer.shape[1] - layer_offset_x
                    layer_img = layer_img[:, :layer_width]

                if layer_offset_y < 0:
                    layer_height = layer_img.shape[0] + layer_offset_y
                    layer_offset_y = 0
                    layer_img = layer_img[-layer_height:, :]
                else:
                    if layer_height + layer_offset_y > buffer.shape[0]:
                        layer_height = buffer.shape[0] - layer_offset_y
                    layer_img = layer_img[:layer_height, :]

                if layer_height > buffer.shape[0]:
                    layer_height = buffer.shape[0]
                    layer_img = layer_img[:layer_height]
                if layer_width > buffer.shape[1]:
                    layer_width = buffer.shape[1]
                    layer_img = layer_img[:, :layer_width]

                # Alpha blend
                premultiplied = layer_type == "vector"

                buffer[
                    layer_offset_y : layer_offset_y + layer_height,
                    layer_offset_x : layer_offset_x + layer_width,
                ] = alpha_blend(
                    buffer[
                        layer_offset_y : layer_offset_y + layer_height,
                        layer_offset_x : layer_offset_x + layer_width,
                    ],
                    layer_img,
                    premultiplied=premultiplied,
                )

        return buffer
