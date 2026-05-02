from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
from PIL import Image

import logging

from clip_tools.blending import composite_layer
from clip_tools.constants import DEBUG, LayerComposite, LayerFolderBit, LayerLockBit
from clip_tools.structs import (
    process_text_attributes,
    process_resizable_image_attributes,
)
from clip_tools.types import LayerEntry, LayerRecord
from clip_tools.utils import (
    calculate_homography,
    backward_mapping,
    alpha_blend,
    arr_to_pil,
)

logger = logging.getLogger(__name__)


class ClipLayer:
    def __init__(
        self,
        record: pd.DataFrame,
        idx: int,
        raster: Dict[int, LayerEntry],
        canvas_size: Tuple[int, int],
        layer_map: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            record (pd.DataFrame): Layer dataframe.
            idx (int): Index of the layer in the layer dataframe.
            raster (dict): Map from layer_id to the raster image.
            canvas_size (Tuple[int, int]): Size of the canvas.
            layer_map (Optional[Dict[int, int]]): Map from layer_id to index in the layer dataframe.
        """
        self._record = record
        self._idx = idx
        self._raster = raster
        self._canvas_size = canvas_size

        self._text_attrs = None
        self._resizable_image_attrs = None
        self._composited = None

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
    def metadata(self) -> LayerRecord:
        """Typed view of this layer's row in the dataframe.

        Built fresh on each access — the dataframe stays the source of truth
        for writes (visibility/opacity setters), so the snapshot doesn't go
        stale.
        """
        return LayerRecord.from_row(self._record.loc[self._idx])

    @property
    def layer_id(self) -> int:
        return int(self._record["MainId"].loc[self._idx])

    @property
    def name(self) -> str:
        return self.metadata.layer_name

    @property
    def layer_type(self) -> str:
        if self.layer_id in self._raster:
            return self._raster[self.layer_id].type
        else:
            return "raster"

    @property
    def parent_id(self) -> int:
        return self.metadata.parent_layer

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
        return self.metadata.layer_visibility != 0

    @visible.setter
    def visible(self, value: bool) -> None:
        self._record.loc[self._idx, "LayerVisibility"] = 1 if value else 0

    @property
    def opacity(self) -> float:
        return self.metadata.layer_opacity / 256.0

    @opacity.setter
    def opacity(self, value: float) -> None:
        ## TODO: WILL LIKELY FAIL
        self._record.loc[self._idx, "LayerOpacity"] = int(value * 256)

    @property
    def size(self) -> Tuple[int, int]:
        """Will return width, height."""
        return self._canvas_size[::-1][0], self._canvas_size[::-1][1]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        meta = self.metadata

        if meta.text_layer_type is not None and meta.text_layer_attributes is not None:
            if DEBUG:
                if not os.path.exists("temp/text_attributes"):
                    os.makedirs("temp/text_attributes")
                with open(
                    f"temp/text_attributes/{self.layer_id}_text_attribute.binary",
                    mode="wb",
                ) as f:
                    f.write(meta.text_layer_attributes)

            try:
                if self._text_attrs is None:
                    self._text_attrs = process_text_attributes(
                        meta.text_layer_attributes
                    )

                attr_ds = self._text_attrs
                layer_offset_x = meta.layer_offset_x + attr_ds.general_offset_x
                layer_offset_y = meta.layer_offset_y + attr_ds.general_offset_y
            except Exception:
                print(
                    f"WARNING: Text attribute layer load failed for layer_id: {self.layer_id}"
                )
                layer_offset_x = meta.layer_offset_x
                layer_offset_y = meta.layer_offset_y
        else:
            layer_offset_x = 0
            layer_offset_y = 0

        # Once the layer's offscreen has been refreshed, render-offscreen
        # offsets are already baked into the offscreen and we shouldn't add
        # them again. DrawToRenderOffscreenType is the dirty/applied flag.
        if meta.draw_to_render_offscreen_type is None:
            layer_offset_x += meta.layer_render_offscr_offset_x
            layer_offset_y += meta.layer_render_offscr_offset_y

        layer_offset_x += meta.layer_offset_x
        layer_offset_y += meta.layer_offset_y

        return (layer_offset_x, layer_offset_y, 0, 0)

    def is_group(self) -> bool:
        return bool(self.metadata.layer_folder & LayerFolderBit.IS_FOLDER)

    def __iter__(self) -> Iterator[ClipLayer]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def composite(self, prepended_layers=[]) -> Optional[Image.Image]:
        entry = self._raster.get(self.layer_id)
        meta = self.metadata

        if entry is not None and entry.type in ("raster", "vector"):
            composited = entry.image
        elif self._composited is not None:
            composited = self._composited
        elif meta.draw_color_enable == 1.0:
            scale = 255.0 / (2**32 - 1)
            red = scale * (meta.draw_color_main_red or 0.0)
            green = scale * (meta.draw_color_main_green or 0.0)
            blue = scale * (meta.draw_color_main_blue or 0.0)
            bg_color = np.array([red, green, blue, 255], dtype=np.uint8)
            composited = np.zeros((*self._canvas_size, 4), dtype=np.uint8)
            composited[..., :4] = bg_color
        else:
            logger.debug("Composit children:", self.children_names)

            if len(self.children) > 0:
                layers_to_composit = prepended_layers + self.children
                composited = self.composit_layers(layers_to_composit)
                # ClipStudio sometimes flattens group strokes into the group's
                # own cached raster and leaves child leaves empty. If child
                # composition produced nothing, fall back to the cached group.
                if (
                    entry is not None
                    and entry.type == "group"
                    and composited.shape[-1] >= 4
                    and not composited[..., 3].any()
                ):
                    composited = entry.image
            elif entry is not None and entry.type == "group":
                composited = entry.image
            else:
                return None

        self._composited = composited
        return arr_to_pil(composited)

    def composit_layers(self, layer_list: List[ClipLayer]) -> np.ndarray:
        """
        layer_list is a list of tuples (id, image)
        """

        def layer_composite(meta: LayerRecord) -> LayerComposite:
            try:
                return LayerComposite(meta.layer_composite)
            except ValueError:
                return LayerComposite.NORMAL

        def layer_alpha_lock(meta: LayerRecord) -> bool:
            return bool(meta.layer_lock & LayerLockBit.ALPHA)

        def unpremultiply_rgba(arr: np.ndarray) -> np.ndarray:
            if arr.shape[-1] < 4:
                return arr
            a = arr[..., 3:4].astype(np.float32) / 255.0
            safe_a = np.where(a == 0, 1.0, a)
            rgb = np.clip(arr[..., :3].astype(np.float32) / safe_a, 0, 255)
            out = arr.copy()
            out[..., :3] = np.where(a > 0, rgb, 0).round().astype(np.uint8)
            return out

        buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

        buffer[..., :3] = 255

        for i, layer in enumerate(layer_list):
            layer_id = layer.layer_id
            layer_type = layer.layer_type
            layer_img = np.array(layer.composite())

            if len(layer_img.shape) == 0:
                continue

            meta = layer.metadata

            if meta.resizable_image_info is not None:
                if DEBUG:
                    if not os.path.exists("temp/resizable_image_info"):
                        os.makedirs("temp/resizable_image_info")
                    with open(
                        f"temp/resizable_image_info/{layer_id}_resizable_image_info.binary",
                        mode="wb",
                    ) as f:
                        f.write(meta.resizable_image_info)

                layer._resizable_image_attrs = process_resizable_image_attributes(
                    meta.resizable_image_info
                )

                layer_buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

                source_coords = layer._resizable_image_attrs.source_coords
                polygon_coords = layer._resizable_image_attrs.polygon_coords
                transform = calculate_homography(source_coords, polygon_coords)
                layer_buffer = backward_mapping(
                    transform, layer_img, layer_buffer, polygon_coords
                )
                mode = layer_composite(meta)
                if mode == LayerComposite.NORMAL:
                    buffer = alpha_blend(buffer, layer_buffer, premultiplied=False)
                else:
                    buffer = composite_layer(
                        buffer,
                        layer_buffer,
                        mode,
                        preserve_transparency=layer_alpha_lock(meta),
                    )

            else:
                if layer_img is None:
                    logger.debug(f"Skipping None layer idx: {i}, id: {layer_id}")
                    continue

                if meta.layer_visibility == 0:
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

                premultiplied = layer_type == "vector"
                mode = layer_composite(meta)
                base_slice = buffer[
                    layer_offset_y : layer_offset_y + layer_height,
                    layer_offset_x : layer_offset_x + layer_width,
                ]
                if mode == LayerComposite.NORMAL:
                    composited = alpha_blend(
                        base_slice, layer_img, premultiplied=premultiplied
                    )
                else:
                    blend_input = (
                        unpremultiply_rgba(layer_img) if premultiplied else layer_img
                    )
                    composited = composite_layer(
                        base_slice,
                        blend_input,
                        mode,
                        preserve_transparency=layer_alpha_lock(meta),
                    )
                buffer[
                    layer_offset_y : layer_offset_y + layer_height,
                    layer_offset_x : layer_offset_x + layer_width,
                ] = composited

        return buffer
