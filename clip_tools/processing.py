import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from clip_tools.constants import DEBUG
from clip_tools.structs import process_layer_blocks
from clip_tools.utils import arr_to_pil


logger = logging.getLogger(__name__)


def build_external_id_map(dfs: Dict[str, pd.DataFrame]) -> dict:
    """Map each external chunk id to its owning table and column."""
    external_id_map: dict = {}
    for _, row in dfs["ExternalTableAndColumnName"].iterrows():
        if row["TableName"] not in dfs:
            continue
        external_ids = dfs[row["TableName"]][row["ColumnName"]]
        for external_id in external_ids:
            external_id_str = external_id.decode("UTF-8")
            external_id_map[external_id_str] = {
                "table_name": row["TableName"],
                "column_name": row["ColumnName"],
                "found": False,
            }
    return external_id_map


def augment_layer_df(layer_df: pd.DataFrame) -> pd.DataFrame:
    """Add ParentLayer and Prefix columns to layer_df for easier traversal."""
    layer_map = {row["MainId"]: index for index, row in layer_df.iterrows()}

    layer_df["ParentLayer"] = 0

    for _, row in layer_df.iterrows():
        if row["LayerFolder"] not in [1, 17]:
            continue
        if row["LayerFirstChildIndex"] == 0:
            continue
        layer_df.loc[layer_map[row["LayerFirstChildIndex"]], "ParentLayer"] = row[
            "MainId"
        ]
        child_layer = layer_df.loc[layer_map[row["LayerFirstChildIndex"]]]
        while child_layer["LayerNextIndex"] != 0:
            layer_df.loc[layer_map[child_layer["LayerNextIndex"]], "ParentLayer"] = row[
                "MainId"
            ]
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


def _save_debug_layer_image(arr: np.ndarray, name: str, key: str, mode: str) -> None:
    temp_folder = f"temp/{name}/external"
    os.makedirs(temp_folder, exist_ok=True)
    if mode == "raster":
        arr_to_pil(arr).save(os.path.join(temp_folder, f"{key}.png"))
    else:
        Image.fromarray(arr, "RGBA").save(os.path.join(temp_folder, f"{key}.png"))


def process_clip_data(
    name: str,
    clip_data: dict,
    dfs: Dict[str, pd.DataFrame],
    layer_df: pd.DataFrame,
    external_id_map: dict,
) -> Tuple[Dict[int, dict], List[dict]]:
    """Classify processed chunks into raster/vector layers and an auxiliary bucket."""
    raster_dict: Dict[int, dict] = {}
    auxillary_list: List[dict] = []
    processed: set = set()

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
                _save_debug_layer_image(processed_layer_arr, name, key, "raster")

            layer_id = offscreen["LayerId"]

            if layer_id == 0:
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
            processed.add(layer_id)

            logger.debug(
                f"Processing layer: {layer_name} in folder: {layer_prefix} with ID: {layer_id}"
            )
            raster_dict[layer_id] = {"type": "raster", "image": processed_layer_arr}
            del blocks

        elif isinstance(value, np.ndarray):
            if DEBUG:
                _save_debug_layer_image(value, name, key, "vector")

            vector_object = dfs["VectorObjectList"][
                dfs["VectorObjectList"]["VectorData"] == key.encode("ascii")
            ].iloc[0]
            layer_id = vector_object["LayerId"]

            if layer_id not in layer_df["MainId"].values:
                logger.debug(f"Skipping invalid layer: {layer_id}")
                auxillary_list.append({"type": "invalid", "image": value})
                continue

            raster_dict[layer_id] = {"type": "vector", "image": value}

    return raster_dict, auxillary_list


def dump_dfs_csv(dfs: Dict[str, pd.DataFrame], base_name: str) -> None:
    """Debug helper: dump all SQLite tables to CSV under temp/<base_name>/csvs/."""
    csv_dir = f"temp/{base_name}/csvs"
    os.makedirs(csv_dir, exist_ok=True)
    for key, df in dfs.items():
        df.to_csv(f"{csv_dir}/{key}.csv", index=False)
