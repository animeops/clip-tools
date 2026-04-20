import logging
import os

from clip_tools.api.clip_layer import ClipLayer
from clip_tools.constants import DEBUG
from clip_tools.io import load_sqlite, split_clip_binary
from clip_tools.processing import (
    augment_layer_df,
    build_external_id_map,
    dump_dfs_csv,
    process_clip_data,
)
from clip_tools.structs import process_chunk_binary

logger = logging.getLogger(__name__)


class ClipImage(ClipLayer):
    @classmethod
    def open(cls, path: str) -> ClipLayer:
        base_name = os.path.splitext(os.path.basename(path))[0]

        with open(path, "rb") as f:
            clip_str = f.read()

        clip_binary, sqlite_binary = split_clip_binary(clip_str)
        dfs = load_sqlite(sqlite_binary)
        logger.debug("Loaded CLIP SQLite database")

        if DEBUG:
            dump_dfs_csv(dfs, base_name)

        external_id_map = build_external_id_map(dfs)
        canvas_size = (
            int(dfs["Canvas"]["CanvasHeight"].iloc[0]),
            int(dfs["Canvas"]["CanvasWidth"].iloc[0]),
        )
        brush_style = dfs.get("BrushStyle")

        clip_data, _num_skipped_chunks = process_chunk_binary(
            clip_binary, canvas_size, external_id_map, brush_style
        )

        for key, value in external_id_map.items():
            if not value.found:
                logger.debug(
                    f"External ID: {key} not found in {value.table_name} {value.column_name}"
                )

        logger.debug("Processed CLIP binary data")

        record = augment_layer_df(dfs["Layer"])
        raster, _auxillary_list = process_clip_data(
            base_name, clip_data, dfs, record, external_id_map
        )

        return ClipLayer(record, 0, raster, canvas_size)
