from __future__ import annotations

import glob
import os
import shutil
import unicodedata
from typing import List, Tuple

import numpy as np
import tqdm
from PIL import Image
from psd_tools import PSDImage

from clip_tools.api.clip_image import ClipImage
from clip_tools.constants import DEBUG


EDIT_TYPE_FOLDER = False


def fullwidth_to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def check_if_genga_default(text: str) -> bool:
    return (
        fullwidth_to_halfwidth(text).lower() == "g"
        or fullwidth_to_halfwidth(text).lower() == "gen"
        or fullwidth_to_halfwidth(text).lower() == "genga"
        or fullwidth_to_halfwidth(text).lower() == "2gen"
        or text.startswith("原画")
        or "原" in text
    )


def check_if_any_visible_children(folder) -> bool:
    for child in folder:
        if child.visible:
            return True
    return False


class FileProcessor:
    def __init__(self, path: str):
        if not isinstance(path, str):
            path = path.name

        self.path = path

        # Find if there is a .tdts file in the same folder as the path
        self.tdts_paths: List[str] = []

        for file in os.listdir(os.path.dirname(path)):
            if file.endswith(".tdts"):
                self.tdts_paths.append(os.path.join(os.path.dirname(path), file))
            # also find folders with "_ts" in the name
            if "_ts_" in file:
                self.tdts_paths.append(os.path.join(os.path.dirname(path), file))
            if "sheet" in file:
                self.tdts_paths.append(os.path.join(os.path.dirname(path), file))

        if path.endswith(".psd"):
            self.root = PSDImage.open(path)
        elif path.endswith(".clip"):
            self.root = ClipImage.open(path)
        else:
            raise ValueError("Unsupported file format")

    def save_image(self, image: Image.Image, path: str) -> None:
        if path.endswith(".png"):
            image.save(path)
        elif path.endswith(".jpg") or path.endswith(".jpeg"):
            jpeg_image = image.convert("RGB")
            jpeg_image.save(path, quality=95)
        else:
            raise ValueError("Unsupported file format")

    def export(self, ext: str = "png", bg_opacity: int = 255) -> Tuple[str, List[str]]:
        put_lo_paper_in_main = False

        assert ext in ["png", "jpg", "jpeg"]

        base_name = os.path.splitext(os.path.basename(self.path))[0]

        genga_type = base_name.split("_")[-1].lower()
        # If genga_type is not G, replace the check_if_genga function
        if genga_type != "g" and genga_type != "gen":

            def check_if_genga(x: str) -> bool:
                return fullwidth_to_halfwidth(x).lower() == genga_type.lower() or (
                    genga_type.lower() == "sa"
                    and fullwidth_to_halfwidth(x).lower() == "作監"
                )
        else:
            check_if_genga = check_if_genga_default

        special_list = ["bg", "cut_info", "camera"]

        temp_path = f"temp/{base_name}"
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        temp_compressed_path = f"temp/compressed/{base_name}"
        if not os.path.exists(temp_compressed_path):
            os.makedirs(temp_compressed_path)

        canvas_shape: List[int] = [0, 0]
        for folder in self.root:
            # take the max
            canvas_shape = list(np.maximum(canvas_shape, folder.size[::-1]))

            if canvas_shape == [0, 0]:
                if folder.is_group():
                    for child in folder:
                        canvas_shape = list(np.maximum(canvas_shape, child.size[::-1]))
                        if child.is_group():
                            for grandchild in child:
                                canvas_shape = list(
                                    np.maximum(canvas_shape, grandchild.size[::-1])
                                )

        if canvas_shape == [0, 0]:
            raise ValueError("Canvas shape is 0, 0")
        white_bg = bg_opacity * np.ones(
            (canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8
        )

        lo_paper_image = None
        tap_image = None
        bg_image = None

        root_folders = self.root

        for folder in self.root:
            folder_name = fullwidth_to_halfwidth(folder.name).lower()
            if folder.is_group():
                for child in folder:
                    child_name = fullwidth_to_halfwidth(child.name).lower()
                    if child.is_group():
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
                            root_folders = list(self.root) + list(folder)

        for folder in root_folders:
            if folder.is_group():
                folder.visible = True

            # Convert name to lowercase
            if folder.name.lower() == "lo_paper":
                lo_paper_image = folder.composite()
                lo_paper_image.info["offset"] = folder.bbox[:2]
            elif folder.name.lower() == "tap":
                tap_image = folder.composite()
                tap_image.info["offset"] = folder.bbox[:2]

        for folder in root_folders:
            folder.visible = True
            if folder.is_group():
                for child in folder:
                    child.visible = True
                    if child.is_group():
                        for grandchild in child:
                            grandchild.visible = True
                            if grandchild.is_group():
                                for grandgrandchild in grandchild:
                                    grandgrandchild.visible = True

        for folder in root_folders:
            folder_name = fullwidth_to_halfwidth(folder.name).lower()
            if folder_name in special_list or folder_name.startswith("bg"):
                composited = Image.fromarray(white_bg.copy())

                if folder.is_group():
                    if folder_name == "camera" or folder_name.startswith("bg"):
                        for child in folder:
                            child_name = fullwidth_to_halfwidth(child.name).lower()
                            is_cam = child_name.startswith("cam")

                            if (
                                child.is_group() or child_name.startswith("book")
                            ) and not is_cam:
                                child_composited = Image.fromarray(white_bg.copy())
                                child_image = child.composite()
                                child_image.info["offset"] = child.bbox[:2]
                                child_composited.paste(
                                    child_image,
                                    child_image.info["offset"],
                                    child_image,
                                )

                                if lo_paper_image is not None:
                                    child_composited.paste(
                                        lo_paper_image,
                                        lo_paper_image.info["offset"],
                                        lo_paper_image,
                                    )
                                if tap_image is not None:
                                    child_composited.paste(
                                        tap_image,
                                        tap_image.info["offset"],
                                        tap_image,
                                    )
                                if child_name.startswith("bg") and child_name != "bg":
                                    self.save_image(
                                        child_composited,
                                        os.path.join(temp_path, f"_{child.name}.{ext}"),
                                    )
                                    child.visible = False
                                elif child_name.startswith("book"):
                                    self.save_image(
                                        child_composited,
                                        os.path.join(
                                            temp_path,
                                            f"_{folder.name}_{child.name}.{ext}",
                                        ),
                                    )
                                    child.visible = True

                            elif is_cam:
                                if child.is_group():
                                    for grandchild in child:
                                        grandchild_composited = Image.fromarray(
                                            white_bg.copy()
                                        )
                                        grandchild_image = grandchild.composite()
                                        grandchild_image.info["offset"] = (
                                            grandchild.bbox[:2]
                                        )
                                        grandchild_composited.paste(
                                            grandchild_image,
                                            grandchild_image.info["offset"],
                                            grandchild_image,
                                        )

                                        if lo_paper_image is not None:
                                            grandchild_composited.paste(
                                                lo_paper_image,
                                                lo_paper_image.info["offset"],
                                                lo_paper_image,
                                            )
                                        if tap_image is not None:
                                            grandchild_composited.paste(
                                                tap_image,
                                                tap_image.info["offset"],
                                                tap_image,
                                            )

                                        self.save_image(
                                            grandchild_composited,
                                            os.path.join(
                                                temp_path,
                                                f"_{child.name}_{grandchild.name}.{ext}",
                                            ),
                                        )
                                        grandchild.visible = False
                                    child.visible = False

                if folder_name == "bg":
                    folder.opacity = 255

                special_image = folder.composite()
                special_image.info["offset"] = folder.bbox[:2]
                composited.paste(
                    special_image, special_image.info["offset"], special_image
                )

                if folder.is_group() and check_if_any_visible_children(folder):
                    if folder_name.startswith("bg"):
                        if folder_name == "bg":
                            bg_image = special_image.copy()

                        if lo_paper_image is not None:
                            composited.paste(
                                lo_paper_image,
                                lo_paper_image.info["offset"],
                                lo_paper_image,
                            )
                        if tap_image is not None:
                            composited.paste(
                                tap_image, tap_image.info["offset"], tap_image
                            )
                        self.save_image(
                            composited,
                            os.path.join(temp_path, f"_{folder.name}.{ext}"),
                        )
                    elif folder_name == "camera":
                        self.save_image(
                            composited, os.path.join(temp_path, f"_PAN.{ext}")
                        )
                    else:
                        self.save_image(
                            composited,
                            os.path.join(temp_path, f"_{folder.name}.{ext}"),
                        )
                elif folder_name.startswith("bg"):
                    folder.opacity = 255
                    special_image = folder.composite()
                    special_image.info["offset"] = folder.bbox[:2]
                    composited.paste(
                        special_image, special_image.info["offset"], special_image
                    )
                    self.save_image(
                        composited, os.path.join(temp_path, f"_{folder.name}.{ext}")
                    )

        bg_images = {}

        for folder in root_folders:
            folder_name = fullwidth_to_halfwidth(folder.name).lower()
            if folder_name.startswith("lo") and not folder_name.startswith("lo_"):
                for child in folder:
                    child_layer_name = fullwidth_to_halfwidth(child.name).lower()
                    if child_layer_name.startswith("paper_"):
                        image = child.composite()
                        image.info["offset"] = child.bbox[:2]
                        bg_images[folder.name] = image

        layout_images = []

        for folder in tqdm.tqdm(root_folders, desc=base_name, disable=DEBUG):
            if DEBUG:
                print(f"Top level folder: {folder.name}")

            folder_name = fullwidth_to_halfwidth(folder.name).lower()

            if (
                folder_name.startswith("lo") and not folder_name.startswith("lo_")
            ) or check_if_genga(folder_name):
                # Check if sublayer is iterable
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
                        if image is None:
                            continue
                        image.info["offset"] = grandchild.bbox[:2]

                        # grandchild_layer_name is in a format like "A4" "B20" "AA30" "A4a"
                        # split the number (the frame idx) from the grandchild_layer_name
                        # In cases like "A4a" it should be "4a"
                        # Find first digit
                        frame_idx = ""
                        for i, c in enumerate(grandchild_layer_name):
                            if c.isdigit():
                                frame_idx = grandchild_layer_name[i:]
                                break

                        if check_if_genga(folder_name) and frame_idx == "1":
                            layout_images.append(image.copy())

                        composited = Image.fromarray(white_bg.copy())

                        if DEBUG:
                            print(
                                f"{child_layer_name}_{grandchild_layer_name}, shape: {composited.size}"
                            )

                        if folder.name in bg_images:
                            composited.paste(
                                bg_images[folder.name],
                                bg_images[folder.name].info["offset"],
                                bg_images[folder.name],
                            )

                        composited.paste(image, image.info["offset"], image)

                        if put_lo_paper_in_main:
                            if lo_paper_image is not None:
                                composited.paste(
                                    lo_paper_image,
                                    lo_paper_image.info["offset"],
                                    lo_paper_image,
                                )
                            if tap_image is not None:
                                composited.paste(
                                    tap_image, tap_image.info["offset"], tap_image
                                )

                        edit_type = folder.name[2:]
                        if check_if_genga(folder_name):
                            self.save_image(
                                composited,
                                os.path.join(
                                    temp_path,
                                    f"{child_layer_name}{grandchild_layer_name}.{ext}",
                                ),
                            )
                        elif folder_name.startswith(
                            "lo"
                        ) and not folder_name.startswith("lo_"):
                            if EDIT_TYPE_FOLDER:
                                folder_path = os.path.join(
                                    temp_path, "_LO", "LO" + edit_type
                                )
                            else:
                                folder_path = os.path.join(temp_path, "_LO")
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            if edit_type == "":
                                self.save_image(
                                    composited,
                                    os.path.join(
                                        folder_path,
                                        f"{child_layer_name}{grandchild_layer_name}.{ext}",
                                    ),
                                )
                            else:
                                self.save_image(
                                    composited,
                                    os.path.join(
                                        folder_path,
                                        f"{child_layer_name}{grandchild_layer_name}_{edit_type}.{ext}",
                                    ),
                                )

        composited = Image.fromarray(white_bg.copy())
        if bg_image is not None:
            composited.paste(bg_image, bg_image.info["offset"], bg_image)
        for image in layout_images:
            composited.paste(image, image.info["offset"], image)
        if lo_paper_image is not None:
            composited.paste(
                lo_paper_image, lo_paper_image.info["offset"], lo_paper_image
            )
        if tap_image is not None:
            composited.paste(tap_image, tap_image.info["offset"], tap_image)

        self.save_image(composited, os.path.join(temp_path, f"_lo.{ext}"))

        # TODO: tdts_paths discovery matches unrelated files (e.g. PDFs with
        # "sheet" in the name) and directories. Disabled until the matcher is
        # tightened and shutil.copy is made directory-safe.
        # for tdts_path in self.tdts_paths:
        #     shutil.copy(tdts_path, temp_path)

        # Compress as zip
        shutil.make_archive(temp_compressed_path, "zip", temp_path)
        return temp_compressed_path + ".zip", glob.glob(f"{temp_path}/*.{ext}")
