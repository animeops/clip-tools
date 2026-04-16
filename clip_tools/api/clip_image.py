import os
from typing import Dict
import sqlite3
import pandas as pd
import numpy as np
import tempfile
import logging
from PIL import Image

from clip_tools.api.clip_layer import ClipLayer
from clip_tools.utils import arr_to_pil
from clip_tools.structs import process_chunk_binary, process_layer_blocks
from clip_tools.constants import DEBUG

logger = logging.getLogger(__name__)


def process_clip_data(
    name: str,
    clip_data: dict,  # TODO: Fix
    dfs: Dict[str, pd.DataFrame],
    layer_df: pd.DataFrame,
    external_id_map: dict,
):
    raster_dict = {}
    auxillary_list = []
    processed = set()

    for key, value in clip_data.items():
        table_name = external_id_map[key]["table_name"]
        column_name = external_id_map[key]["column_name"]

        logger.debug(f"Processing blocks: {key} in {table_name} {column_name}")

        if isinstance(value, dict) and value:
            offscreen = dfs["Offscreen"][
                dfs["Offscreen"]["BlockData"] == key.encode("ascii")
            ].iloc[0]
            blocks = sorted(value.items(), key=lambda x: x[0])
            try:
                processed_layer_arr = process_layer_blocks(blocks, offscreen)
            except Exception:
                logger.error(
                    f"Error processing layer: {key} in {table_name} {column_name}"
                )

            if DEBUG:
                temp_folder = f"temp/{name}/external"
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                processed_layer_image = arr_to_pil(processed_layer_arr)
                processed_layer_image.save(os.path.join(temp_folder, f"{key}.png"))

            layer_id = offscreen["LayerId"]

            if layer_id == 0:
                # Canvas layer
                continue

            if layer_id not in layer_df["MainId"].values:
                logger.debug(f"Skipping invalid layer: {layer_id}")
                auxillary_list.append({"type": "invalid", "image": processed_layer_arr})
                continue

            layer_metadata = layer_df[layer_df["MainId"] == layer_id].iloc[0]
            layer_name = layer_metadata["LayerName"]
            layer_prefix = layer_metadata["Prefix"]
            layer_folder = layer_metadata["LayerFolder"]

            if (
                "LayerClip" in layer_metadata.keys()
                and layer_metadata["LayerClip"] != 0
            ):
                # Skip layers that are clipped
                logger.warning(
                    f"WARNING: Skipping clipped layer: {layer_name} in folder: {layer_prefix}"
                )
                auxillary_list.append({"type": "clipped", "image": processed_layer_arr})
                continue

            if layer_folder != 0:
                auxillary_list.append({"type": "group", "image": processed_layer_arr})
                continue

            if offscreen["MainId"] in dfs["MipmapInfo"]["Offscreen"].values:
                mipmapinfo = dfs["MipmapInfo"][
                    dfs["MipmapInfo"]["Offscreen"] == offscreen["MainId"]
                ].iloc[0]

                if mipmapinfo["ThisScale"] != 100.0:
                    # Skip mipmaps
                    logger.debug(
                        f"Skipping mipmap: {layer_name} in folder: {layer_prefix}"
                    )
                    auxillary_list.append(
                        {"type": "mipmap", "image": processed_layer_arr}
                    )
                    continue
            else:
                auxillary_list.append({"type": "other", "image": processed_layer_arr})
                continue

            if layer_id in processed:
                raise Exception(f"Layer {layer_id} already processed")
            else:
                processed.add(layer_id)

            logger.debug(
                f"Processing layer: {layer_name} in folder: {layer_prefix} with ID: {layer_id}"
            )
            raster_dict[layer_id] = {"type": "raster", "image": processed_layer_arr}
            del blocks

        elif isinstance(value, np.ndarray):
            if DEBUG:
                temp_folder = f"temp/{name}/external"
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                processed_layer_image = Image.fromarray(value, "RGBA")
                processed_layer_image.save(os.path.join(temp_folder, f"{key}.png"))

            vector_object = dfs["VectorObjectList"][
                dfs["VectorObjectList"]["VectorData"] == key.encode("ascii")
            ].iloc[0]
            layer_id = vector_object["LayerId"]

            if layer_id not in layer_df["MainId"].values:
                # Skip invalid layers
                print(f"Skipping invalid layer: {layer_id}")
                auxillary_list.append({"type": "invalid", "image": value})
                continue

            raster_dict[layer_id] = {"type": "vector", "image": value}
    return raster_dict, auxillary_list


def augment_layer_df(layer_df):
    """Add new columns to layer_df for easier processing."""
    # Make map from MainId (a column in layer_df to index
    layer_map = {}
    for index, row in layer_df.iterrows():
        layer_map[row["MainId"]] = index

    # Add column called "ParentLayer" to layer_df and fill with 0s
    layer_df["ParentLayer"] = 0

    # Iterate through rows of layer_df
    for _, row in layer_df.iterrows():
        if row["LayerFolder"] in [1, 17]:
            if row["LayerFirstChildIndex"] == 0:
                continue
            layer_df.loc[layer_map[row["LayerFirstChildIndex"]], "ParentLayer"] = row[
                "MainId"
            ]
            child_layer = layer_df.loc[layer_map[row["LayerFirstChildIndex"]]]
            while child_layer["LayerNextIndex"] != 0:
                layer_df.loc[
                    layer_map[child_layer["LayerNextIndex"]], "ParentLayer"
                ] = row["MainId"]
                child_layer = layer_df.loc[layer_map[child_layer["LayerNextIndex"]]]

    layer_df["Prefix"] = [[] for _ in range(len(layer_df))]

    for idx, row in layer_df.iterrows():
        if row["ParentLayer"] == 0 or row["ParentLayer"] == layer_df.loc[0]["MainId"]:
            continue

        parent_layer = layer_df.loc[layer_map[row["ParentLayer"]]]
        layer_df.loc[idx, "Prefix"].insert(0, parent_layer["LayerName"])

        while (
            parent_layer["ParentLayer"] != 0
            and parent_layer["ParentLayer"] != layer_df.loc[0]["MainId"]
        ):
            parent_layer = layer_df.loc[layer_map[parent_layer["ParentLayer"]]]
            layer_df.loc[idx, "Prefix"].insert(0, parent_layer["LayerName"])

    return layer_df


def load_sqlite(sqlite_binary_str):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sqlite_binary_str)
        sqlite_path = f.name

    with sqlite3.connect(sqlite_path) as connect:
        cursor = connect.cursor()
        # vcursor.execute("SELECT name FROM sqlite_master;")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        dfs = {}

        for table in tables:
            table_name = table[0]
            df = pd.read_sql_query(f"SELECT * from {table_name}", connect)
            dfs[table_name] = df

    return dfs


class ClipImage(ClipLayer):
    @classmethod
    def open(cls, path: str) -> ClipLayer:
        base_name = os.path.splitext(os.path.basename(path))[0]

        with open(path, mode="rb") as f:
            clip_str = f.read()

            search_text = "SQLite format 3"
            search_text_bytes = search_text.encode("utf-8")

        find_index = clip_str.find(search_text_bytes)

        if find_index <= 0:
            raise Exception("Invalid CLIP Studio Paint file")

        if 0 <= find_index:
            sqlite_binary_str = clip_str[find_index:]
            clip_binary_str = clip_str[:find_index]
        else:
            raise Exception("Invalid CLIP Studio Paint file")

        dfs = load_sqlite(sqlite_binary_str)
        logger.debug("Loaded CLIP SQLite database")

        if DEBUG:
            # Save each table inside dfs (which is a dict of dataframes) but not as csv
            for key, value in dfs.items():
                # Make paths
                if not os.path.exists(f"temp/{base_name}"):
                    os.makedirs(f"temp/{base_name}")
                if not os.path.exists(f"temp/{base_name}/csvs"):
                    os.makedirs(f"temp/{base_name}/csvs")

                # Dont save as csv
                dfs[key].to_csv(f"temp/{base_name}/csvs/{key}.csv", index=False)

        # Create mapping from externalid to table and column name
        external_id_map = {}
        for index, row in dfs["ExternalTableAndColumnName"].iterrows():
            if row["TableName"] in dfs:
                external_ids = dfs[row["TableName"]][row["ColumnName"]]
                for external_id in external_ids:
                    external_id_str = external_id.decode("UTF-8")
                    external_id_map[external_id_str] = {}
                    external_id_map[external_id_str]["table_name"] = row["TableName"]
                    external_id_map[external_id_str]["column_name"] = row["ColumnName"]
                    external_id_map[external_id_str]["found"] = False

        _external_id_map = external_id_map

        _canvas_size = (
            int(dfs["Canvas"]["CanvasHeight"].iloc[0]),
            int(dfs["Canvas"]["CanvasWidth"].iloc[0]),
        )

        brush_style = None
        if "BrushStyle" in dfs:
            brush_style = dfs["BrushStyle"]

        clip_data, num_skipped_chunks = process_chunk_binary(
            clip_binary_str, _canvas_size, _external_id_map, brush_style
        )

        # Check if all external ids are found
        for key, value in _external_id_map.items():
            if not value["found"]:
                logger.debug(
                    f"External ID: {key} not found in {value['table_name']} {value['column_name']}"
                )

        # if DEBUG:
        # for key in self.num_skipped_chunks.keys():
        #     ascii_key = key.encode('ascii')
        #     if ascii_key in self.dfs["Offscreen"]["BlockData"].values:
        #         offscreen = self.dfs["Offscreen"][self.dfs["Offscreen"]["BlockData"] == key.encode('ascii')].iloc[0]
        #         layer_id = offscreen["LayerId"]
        #         print(f"Layer ID: {layer_id}, chunks: {len(self.clip_data[key])}, skipped chunks: {self.num_skipped_chunks[key]}")

        logger.debug("Processed CLIP binary data")

        _record = augment_layer_df(dfs["Layer"])

        _raster, auxillary_list = process_clip_data(
            base_name, clip_data, dfs, _record, external_id_map
        )

        return ClipLayer(_record, 0, _raster, _canvas_size)
