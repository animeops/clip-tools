from __future__ import annotations
import os

import numpy as np
from PIL import Image
import tqdm

from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer, Group

from clip_tools.api.clip_image import ClipImage
from clip_tools.api.clip_layer import ClipLayer
from .constants import DEBUG

import unicodedata
import tempfile
from typing import List, Tuple, Union, Optional


def fullwidth_to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def process_name(name: str) -> str:
    return fullwidth_to_halfwidth(name).lower()


def check_if_genga(text: str) -> bool:
    return fullwidth_to_halfwidth(text).lower() == "g" or text.startswith("原画")


def check_if_cam(text: str) -> bool:
    possible_prefixes = ["cam", "camera", "pan"]
    return any([process_name(text).startswith(prefix) for prefix in possible_prefixes])


def check_if_bg(text: str) -> bool:
    possible_name = ["bg", "background", "背景"]
    return process_name(text) in possible_name


def check_if_lo(text: str) -> bool:
    possible_name = ["lo", "layout", "レイアウト"]
    possible_prefixes = [
        "lo",
    ]
    return process_name(text) in possible_name or any(
        [process_name(text).startswith(prefix) for prefix in possible_prefixes]
    )


def check_if_backing_paper(text: str) -> bool:
    possible_prefixes = [
        "paper_",
    ]
    return any([process_name(text).startswith(prefix) for prefix in possible_prefixes])


def check_if_bg_prefix(text: str) -> Tuple[bool, Optional[str]]:
    """
    Also returns the suffix
    """
    possible_prefixes = ["bg", "background", "背景"]
    for prefix in possible_prefixes:
        if process_name(text).startswith(prefix):
            suffix = text[len(prefix) :]
            # remove any underscores
            suffix = suffix.lstrip("_")
            return True, suffix
    return False, None


def check_comment_type(text: str) -> Optional[str]:
    """
    Will detect strings like
    LOEN -> EN
    LO_SA -> SA
    _ensyu -> EN
    _sa -> SA

    A comment can be EN, SA, SS. Usually, this is prefixed in the name...
    and it should be followed by a character like a space, an underscore, a dash,
    or "syu". Determine the comment type.
    """
    possible_prefixes = ["en", "sa", "ss"]

    # Remove any underscores, dashes, or spaces that prefixes the text
    processed_text = text.lstrip("_").lstrip("-").lstrip(" ")
    processed_text = process_name(processed_text)

    prefix_name = None
    for prefix in possible_prefixes:
        if processed_text == prefix:
            return prefix
        elif processed_text.startswith(prefix):
            prefix_name = prefix
            suffix = text[len(prefix_name) :]
            if suffix[0] in [" ", "_", "-", "syu", ""]:
                return prefix
        elif prefix in processed_text:
            if processed_text.endswith(prefix) or processed_text.endswith(
                prefix + "syu"
            ):
                return prefix


def check_if_any_visible_children(folder: Union[ClipLayer, Group, PSDImage]) -> bool:
    """
    Check if any of the children are visible
    """
    for child in folder:
        if child.visible:
            return True
    return False


def find_canvas_size(
    cutfile: Union[ClipLayer, Group, PSDImage], canvas_shape: Tuple[int, int] = (0, 0)
) -> Tuple[int, int]:
    """
    Find the canvas size of the cutfile, recursively
    """
    canvas_shape = (0, 0)
    for folder in cutfile:
        # take the max
        max_shape = np.maximum(folder.size[::-1], canvas_shape)
        canvas_shape = (max_shape[0], max_shape[1])

        if canvas_shape == [0, 0] and folder.is_group():
            canvas_shape = find_canvas_size(folder)
    return canvas_shape


def set_all_visible(cutfile: Union[ClipLayer, Group, PSDImage]) -> None:
    """
    Set all layers to visible
    """
    for folder in cutfile:
        folder.visible = True
        if folder.is_group():
            set_all_visible(folder)


def add_alternate_roots(
    cutfile: Union[ClipLayer, Group, PSDImage],
) -> List[Union[ClipLayer, Group]]:
    """
    Sometimes the cutfile is organized in a way that some folders are in the wrong place.

    So, add anything that starts with "LO" or "BG" or anything that seems like a GENGA folder to the search path
    """
    root_folders = list(cutfile)
    for folder in cutfile:
        folder_name = fullwidth_to_halfwidth(folder.name).lower()
        if folder.is_group():
            for child in folder:
                child_name = fullwidth_to_halfwidth(child.name).lower()
                if child.is_group():
                    # print(f"Checking {folder_name} {child_name}")
                    # if check_if_genga(child.name) or child_name.startswith("lo") or child_name.startswith("bg"):
                    # TODO: This is very fragile, need to find a better way to detect when things are organized in weird layers
                    # TODO: Maybe use the fact that GENGA is not in LO
                    if (
                        child_name.startswith("lo")
                        or (child_name == "bg" and folder_name != "bg")
                        or (
                            check_if_genga(child_name)
                            and not folder_name.startswith("lo")
                            and not folder_name.startswith("g")
                        )
                    ):
                        root_folders += list(folder)
    return root_folders


def find_lo_paper(
    root_folders: List[Union[ClipLayer, Group]],
) -> Optional[Union[Group, PSDImage, ClipLayer]]:
    """
    Finds the lo_paper folder in the root folders.
    """
    possible_names = [
        "lo_paper",
    ]

    for folder in root_folders:
        folder_name = process_name(folder.name)
        if folder_name in possible_names:
            return folder


def find_tap(
    root_folders: List[Union[ClipLayer, Group]],
) -> Optional[Union[Group, PSDImage, ClipLayer]]:
    """
    Finds the tap folder in the root folders.
    """
    possible_names = [
        "tap",
    ]

    for folder in root_folders:
        folder_name = process_name(folder.name)
        if folder_name in possible_names:
            return folder


def composite_and_paste(
    layer: Union[ClipLayer, Group, PixelLayer, PSDImage], canvas_shape: Tuple[int, int]
) -> Optional[Image.Image]:
    """
    Composite the layer and paste it on an empty background
    """
    white_bg = 0 * np.ones((canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8)
    composited = Image.fromarray(white_bg.copy())
    image = layer.composite()
    if image is None:
        return None
    image.info["offset"] = layer.bbox[:2]
    composited.paste(image, image.info["offset"], image)
    return composited


def get_cel_number_from_layer_name(layer_name: str) -> Optional[str]:
    """
    Usually folders are organized like:

    A (folder)
    - A1 (folder)
    - A1a (folder)

    Or like:

    A (folder)
    - 1 (folder)
    - 1a (folder)

    In these cases, cel_name is A and layer_name is A1 or A1a.

    In either case, we need to extract the "digit" like "1" or "1a" from the layer_name.
    Assumes the "digit" starts from a digit.
    """
    for i, c in enumerate(layer_name):
        if c.isdigit():
            return layer_name[i:].strip("_")


class FileProcessor:
    def __init__(self, path: str):
        if not isinstance(path, str):
            path = path.name

        self.path = path

        # Find if there is a .tdts file in the same folder as the path
        self.tdts_paths = []

        for file in os.listdir(os.path.dirname(path)):
            if file.endswith(".tdts"):
                self.tdts_paths.append(os.path.join(os.path.dirname(path), file))
            # also find folders with "_ts" in the name
            if "_ts_" in file:
                self.tdts_paths.append(os.path.join(os.path.dirname(path), file))

        if path.endswith(".psd"):
            self.root = PSDImage.open(path)
        elif path.endswith(".clip"):
            self.root = ClipImage.open(path)
        else:
            raise ValueError("Unsupported file format")

    def export(self, path=None, ext="png"):
        assert ext in ["png", "jpg", "jpeg"]

        base_name = os.path.splitext(os.path.basename(self.path))[0]

        special_list = ["bg", "cut_info", "camera"]

        if path:
            temp_path = path
        else:
            temp_path = tempfile.mkdtemp()

        canvas_shape = find_canvas_size(self.root)

        if canvas_shape == [0, 0]:
            raise ValueError("Canvas shape is 0, 0")

        white_bg = 0 * np.ones((canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8)

        lo_paper_image = None
        tap_image = None
        bg_image = None

        set_all_visible(self.root)
        root_folders = add_alternate_roots(self.root)
        lo_paper_image = find_lo_paper(root_folders)
        tap_image = find_tap(root_folders)

        os.makedirs(os.path.join(temp_path, "meta"), exist_ok=True)
        if lo_paper_image is not None:
            lo_paper_composited = composite_and_paste(lo_paper_image, canvas_shape)
            if lo_paper_composited is not None:
                lo_paper_composited.save(
                    os.path.join(temp_path, "meta", "lo_paper.png")
                )
        if tap_image is not None:
            tap_composited = composite_and_paste(tap_image, canvas_shape)
            if tap_composited is not None:
                tap_composited.save(os.path.join(temp_path, "meta", "tap.png"))

        os.makedirs(os.path.join(temp_path, "bg"), exist_ok=True)
        os.makedirs(os.path.join(temp_path, "bg", "book"), exist_ok=True)
        os.makedirs(os.path.join(temp_path, "cam"), exist_ok=True)
        os.makedirs(os.path.join(temp_path, "backing_paper"), exist_ok=True)

        for folder in root_folders:
            # Look through root folders, and write out special folders
            # (bg, book, cam, etc)
            folder_name = process_name(folder.name)
            folder.opacity = 255

            if check_if_bg(folder_name):
                if folder.is_group():
                    for child in folder:
                        child_name = process_name(child.name)
                        if child.is_group():
                            child_composited = composite_and_paste(child, canvas_shape)
                            if child_composited is not None:
                                child_composited.save(
                                    os.path.join(
                                        temp_path, "bg", f"_{child.name}.{ext}"
                                    )
                                )
                                child.visible = False
                        elif child_name.startswith("book"):
                            child_composited = composite_and_paste(child, canvas_shape)
                            if child_composited is not None:
                                child_composited.save(
                                    os.path.join(
                                        temp_path, "bg", "book", f"_{child.name}.{ext}"
                                    )
                                )
                                child.visible = False
                folder_composited = composite_and_paste(folder, canvas_shape)
                if folder_composited is not None:
                    folder_composited.save(os.path.join(temp_path, "bg", f"bg.{ext}"))

            elif check_if_bg_prefix(folder_name)[0]:
                suffix = check_if_bg_prefix(folder_name)[1]
                folder_composited = composite_and_paste(folder, canvas_shape)
                if folder_composited is not None:
                    folder_composited.save(
                        os.path.join(temp_path, "bg", f"{suffix}.{ext}")
                    )

            elif check_if_cam(folder_name):
                if folder.is_group():
                    for child in folder:
                        if child.is_group():
                            for grandchild in child:
                                grandchild_composited = composite_and_paste(
                                    grandchild, canvas_shape
                                )
                                if grandchild_composited is not None:
                                    grandchild_composited.save(
                                        os.path.join(
                                            temp_path,
                                            "cam",
                                            f"{child.name}_{grandchild.name}.{ext}",
                                        )
                                    )
                                    grandchild.visible = False
                        else:
                            child_composited = composite_and_paste(child, canvas_shape)
                            if child_composited is not None:
                                child_composited.save(
                                    os.path.join(
                                        temp_path, "cam", f"{child.name}.{ext}"
                                    )
                                )
                                child.visible = False
                else:
                    folder_composited = composite_and_paste(folder, canvas_shape)
                    if folder_composited is not None:
                        folder_composited.save(
                            os.path.join(temp_path, "cam", f"pan.{ext}")
                        )

            if check_if_lo(folder_name):
                if folder.is_group():
                    for child in folder:
                        child_name = process_name(child.name)
                        if check_if_backing_paper(child_name):
                            os.makedirs(
                                os.path.join(temp_path, "backing_paper", folder_name),
                                exist_ok=True,
                            )
                            child_composited = composite_and_paste(child, canvas_shape)
                            if child_composited is not None:
                                child_composited.save(
                                    os.path.join(
                                        temp_path,
                                        "backing_paper",
                                        folder_name,
                                        f"{child.name}.{ext}",
                                    )
                                )
                                child.visible = False

        representative_layout = []
        representative_genga = []

        for folder in tqdm.tqdm(root_folders, desc=base_name, disable=DEBUG):
            if DEBUG:
                print(f"Top level folder: {folder.name}")

            folder_name = process_name(folder.name)
            comment_type = check_comment_type(folder_name)

            folder_type = None
            if check_if_lo(folder_name):
                folder_type = "lo"
            elif check_if_genga(folder_name):
                folder_type = "genga"

            if folder_type:
                if not folder.is_group():
                    continue

                for child in tqdm.tqdm(
                    folder,
                    desc=f"Folder: [{folder.name}] Progress",
                    leave=False,
                    disable=DEBUG,
                ):
                    child_layer_name = child.name

                    if not child.is_group():
                        continue

                    for grandchild in tqdm.tqdm(
                        child,
                        desc=f"Layer: [{child.name}] Progress",
                        leave=False,
                        disable=DEBUG,
                    ):
                        grandchild_layer_name = grandchild.name

                        image = grandchild.composite()
                        image.info["offset"] = grandchild.bbox[:2]

                        cel_name = child.name
                        cel_number = get_cel_number_from_layer_name(
                            grandchild_layer_name
                        )

                        if cel_number is None:
                            continue

                        composited = Image.fromarray(white_bg.copy())
                        composited.paste(image, image.info["offset"], image)
                        # Make all white pixels transparent
                        composited = composited.convert("RGBA")
                        data = np.array(composited)

                        mask = (
                            np.abs((data[..., :3] - np.array([255, 255, 255]))).sum(-1)
                            < 10
                        )
                        data[mask, ..., -1] = 0
                        composited = Image.fromarray(data)

                        if DEBUG:
                            print(
                                f"{child_layer_name}_{grandchild_layer_name}, shape: {composited.size}"
                            )

                        if comment_type:
                            # TODO: Maybe change the order of the folder_type and comment_type
                            os.makedirs(
                                os.path.join(
                                    temp_path, "frames", comment_type, folder_type
                                ),
                                exist_ok=True,
                            )
                            composited.save(
                                os.path.join(
                                    temp_path,
                                    "frames",
                                    comment_type,
                                    folder_type,
                                    f"{cel_name}_{cel_number}.{ext}",
                                )
                            )
                        else:
                            # Get representative frame
                            if check_if_genga(folder_name) and cel_number == "1":
                                representative_genga.append(image)
                            if check_if_lo(folder_name) and cel_number == "1":
                                representative_layout.append(image)
                            os.makedirs(
                                os.path.join(temp_path, "frames", folder_type),
                                exist_ok=True,
                            )
                            composited.save(
                                os.path.join(
                                    temp_path,
                                    "frames",
                                    folder_type,
                                    f"{cel_name}_{cel_number}.{ext}",
                                )
                            )

        # Save representative images
        if len(representative_layout) > 0:
            composited = Image.fromarray(white_bg.copy())
            for image in representative_layout:
                composited.paste(image, image.info["offset"], image)
            composited.save(os.path.join(temp_path, f"_layout.{ext}"))
        if len(representative_genga) > 0:
            composited = Image.fromarray(white_bg.copy())
            for image in representative_genga:
                composited.paste(image, image.info["offset"], image)
            composited.save(os.path.join(temp_path, f"_genga.{ext}"))

        return temp_path
