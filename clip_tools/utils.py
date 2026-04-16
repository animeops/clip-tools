import numpy as np
import pandas as pd
from PIL import Image
import cv2
from typing import List, Tuple, Any
import struct


def read_binary_spec(
    binary_str: bytes, spec: struct.Struct, pos: int
) -> Tuple[Any, int]:
    buff = binary_str[pos : pos + spec.size]
    pos += spec.size
    return spec.unpack_from(buff), pos


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr, "L")
    elif arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    elif arr.shape[2] == 2:
        return Image.fromarray(arr, "LA")
    else:
        raise Exception("Unsupported image format")


def alpha_blend(
    background: np.ndarray, foreground: np.ndarray, premultiplied: bool = True
) -> np.ndarray:
    min_height = min(background.shape[0], foreground.shape[0])
    min_width = min(background.shape[1], foreground.shape[1])

    alpha = foreground[:min_height, :min_width, 3:4].astype(np.float32) / 255.0

    if premultiplied:
        blended = (
            foreground[:min_height, :min_width, :3]
            + (1 - alpha) * background[:min_height, :min_width, :3]
        )
        # clamp
        blended = np.clip(blended, 0, 255)
    else:
        blended = (
            alpha * foreground[:min_height, :min_width, :3]
            + (1 - alpha) * background[:min_height, :min_width, :3]
        )
    out_alpha = np.maximum(
        foreground[:min_height, :min_width, 3:4],
        background[:min_height, :min_width, 3:4],
    )
    output = np.concatenate((blended, out_alpha), axis=-1).round().astype(np.uint8)

    return output


def search_df_rows(dfs: pd.DataFrame, search_str: str) -> List[str]:
    matches = []
    for _, df in dfs.items():
        matches.append(
            df[
                df.apply(
                    lambda row: row.astype(str).str.contains(search_str).any(), axis=1
                )
            ]
        )
    return matches


def search_df_columns(dfs: pd.DataFrame, search_str: str) -> List[str]:
    matches = []
    for _, df in dfs.items():
        matches.append(df.filter(like=search_str))
    return matches


def convex_polygon(poly_coords: np.ndarray, image_coords: np.ndarray) -> np.ndarray:
    mask = np.ones_like(image_coords[..., 0]).astype(bool)
    N = poly_coords.shape[0]
    for i in range(N):
        dv = poly_coords[(i + 1) % N] - poly_coords[i]
        winding = (image_coords - poly_coords[i][None]) * (np.flip(dv[None], axis=-1))
        winding = winding[..., 0] - winding[..., 1]
        mask = np.logical_and(mask, (winding > 0))
    return mask


def create_coordinates(h: int, w: int) -> np.ndarray:
    window_x = np.arange(w)
    window_y = np.arange(h)
    coords = np.stack(np.meshgrid(window_x, window_y), axis=-1)
    return coords


def calculate_homography(source, destination) -> np.ndarray:
    H_matrix, _ = cv2.findHomography(destination, source)
    return H_matrix


def backward_mapping(
    transform: np.ndarray,
    source_image: np.ndarray,
    destination_image: np.ndarray,
    polygon_coords: np.ndarray,
) -> np.ndarray:
    h, w, _ = destination_image.shape
    coords = create_coordinates(h, w)
    mask = convex_polygon(polygon_coords[::-1], coords)
    filtered_coords = coords[mask]
    backprojected_coords = backward_projection(transform, filtered_coords)
    interpolated_colors = interpolate_coordinates(backprojected_coords, source_image)
    output_buffer = np.zeros_like(destination_image)
    output_buffer[mask] = interpolated_colors
    return output_buffer


def backward_projection(transform: np.ndarray, coords: np.ndarray) -> np.ndarray:
    # Homogeneous transform
    backprojected_coords = (
        transform @ np.concatenate([coords, np.ones_like(coords[..., 0:1])], axis=-1).T
    )
    # Normalize
    backprojected_coords = (backprojected_coords.T)[:, :2] / (backprojected_coords.T)[
        :, 2:
    ]
    return backprojected_coords


def interpolate_coordinates(coords: np.ndarray, image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    qx = coords[..., 0]
    qy = coords[..., 1]
    qx = np.floor(qx).astype(np.int32)
    qy = np.floor(qy).astype(np.int32)
    return image[qy, qx]
