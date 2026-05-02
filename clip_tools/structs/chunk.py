from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import struct
import zlib
import numpy as np
import pandas as pd
import logging

from clip_tools.utils import read_binary_spec
from clip_tools.constants import DEBUG, BlockMarker, ChunkMagic
from clip_tools.types import ExternalIdEntry
from clip_tools.structs.binc import BincDocument, is_binc, parse_binc

logger = logging.getLogger(__name__)


@dataclass
class ChunkHeader:
    """Decoded `CHNKHead` body."""

    version: int  # observed 256 (=0x100; format v1.0?)
    binary_section_size: int  # = (clip-binary-region length) - 16
    identifier_length: int  # observed 16
    identifier: bytes  # per-file UUID-like
    raw: Optional[bytes] = None  # set instead of the above when body < 40 bytes


def parse_chnk_head_body(body: bytes) -> ChunkHeader:
    """Decode the 40-byte CHNKHead body.

    Layout (big-endian)::

        u64  version
        u64  binary_section_size
        u64  identifier_length
        byte[16] identifier
    """
    if len(body) < 40:
        return ChunkHeader(
            version=0,
            binary_section_size=0,
            identifier_length=0,
            identifier=b"",
            raw=bytes(body),
        )
    version = struct.unpack_from(">Q", body, 0)[0]
    binary_section_size = struct.unpack_from(">Q", body, 8)[0]
    identifier_length = struct.unpack_from(">Q", body, 16)[0]
    identifier = bytes(body[24 : 24 + 16])
    return ChunkHeader(
        version=version,
        binary_section_size=binary_section_size,
        identifier_length=identifier_length,
        identifier=identifier,
    )


def process_chunk_binary(
    clip_binary_str: bytes,
    canvas_shape: Tuple[int, int],
    external_id_map: Dict[str, ExternalIdEntry],
    brush_styles: pd.DataFrame,
) -> Tuple[
    Dict[str, Union[Dict[int, bytes], bytes, BincDocument]],
    Dict[str, int],
    Optional[ChunkHeader],
    Dict[str, Dict[str, List[int]]],
]:
    """Parse the chunk-stream binary region.

    Returns:
        chunk_dict: external_id → one of:
            - {block_idx: decompressed_bytes}      (raster Offscreen.BlockData)
            - bytes                                 (vector blobs, kept raw)
            - BincDocument                          (Track/3D/timelapse, decoded)
        num_skipped_dict: external_id → count of unrecognized sub-chunks.
        file_header: decoded CHNKHead body (version, identifier, etc.).
        block_metadata: external_id → {"statuses": [int], "checksums": [int]}
            for raster chunks that ship their per-block status/checksum arrays.
    """
    clip_header_spec = struct.Struct(">8sQQ")
    chunk_header_spec = struct.Struct(">8sQ")
    external_header_spec = struct.Struct(">Q40sQ")
    block_test_spec = struct.Struct(">II")
    uint_spec = struct.Struct(">I")
    uint_spec_alt = struct.Struct("<I")  # Little endian
    block_header_spec = struct.Struct(">I12xI")

    chunk_dict: Dict[str, Union[Dict[int, bytes], bytes]] = {}
    num_skipped_dict: Dict[str, int] = {}
    block_metadata: Dict[str, Dict[str, List[int]]] = {}
    file_header: Optional[ChunkHeader] = None
    unknown_bd_ids: Dict[bytes, int] = {}

    num_external_ids = 0

    pos = 0
    data, pos = read_binary_spec(clip_binary_str, clip_header_spec, pos)
    _, filesize, _ = data

    while pos < filesize:
        data, pos = read_binary_spec(clip_binary_str, chunk_header_spec, pos)
        chunk_type, chunk_length = data

        if chunk_type == ChunkMagic.HEADER:
            file_header = parse_chnk_head_body(
                clip_binary_str[pos : pos + chunk_length]
            )
            pos += chunk_length
        elif chunk_type == ChunkMagic.EXTERNAL:
            data, pos = read_binary_spec(clip_binary_str, external_header_spec, pos)

            _, external_id, data_size = data
            external_id_str = external_id.decode("UTF-8")

            num_skipped_dict[external_id_str] = 0

            if external_id_str not in external_id_map:
                raise Exception(
                    f"External ID {external_id_str} not found in CLIP SQLite database"
                )
            else:
                external_id_map[external_id_str].found = True

            chunk_dict[external_id_str] = {}
            data_binary_str = clip_binary_str[pos : pos + data_size]

            num_external_ids += 1

            logger.debug(
                f"Processing external ID #{num_external_ids} with ID {external_id_str}..."
            )

            # Track / 3D model / timelapse externals are stored flat as
            # `[u32 LE size][zlib payload]`, not as BlockData chunks.
            # The decompressed payload is a Celsys binc document.
            if data_size >= 8:
                payload_size_le = struct.unpack_from("<I", data_binary_str, 0)[0]
                if payload_size_le == data_size - 4 and data_binary_str[4:6] in (
                    b"\x78\x01",
                    b"\x78\x9c",
                    b"\x78\xda",
                ):
                    try:
                        decompressed = zlib.decompress(data_binary_str[4:data_size])
                    except zlib.error:
                        decompressed = b""
                    if decompressed and is_binc(decompressed):
                        chunk_dict[external_id_str] = parse_binc(decompressed)
                    elif decompressed:
                        chunk_dict[external_id_str] = decompressed
                    else:
                        chunk_dict[external_id_str] = bytes(data_binary_str[:data_size])
                    pos += data_size
                    continue

            chunk_pos = 0
            while chunk_pos < data_size:
                data, chunk_pos = read_binary_spec(
                    data_binary_str, block_test_spec, chunk_pos
                )

                if data == (88, 72):
                    chunk_pos -= 8
                    # Vector blob: store raw bytes; rasterization happens later
                    # once brush patterns (other clip_data entries) are loaded.
                    chunk_dict[external_id_str] = bytes(data_binary_str[:data_size])
                    chunk_pos = data_size
                    continue

                if (
                    data[1]
                    == uint_spec.unpack(BlockMarker.BLOCK_DATA_BEGIN[: uint_spec.size])[
                        0
                    ]
                ):
                    str_length = data[0]
                    chunk_pos -= uint_spec.size
                else:
                    str_length = data[1]
                    data_length = data[0]

                bd_id = data_binary_str[chunk_pos : chunk_pos + (str_length * 2)]
                chunk_pos += str_length * 2

                if bd_id == BlockMarker.BLOCK_DATA_BEGIN:
                    data, chunk_pos = read_binary_spec(
                        data_binary_str, block_header_spec, chunk_pos
                    )
                    block_idx, has_content = data
                    if has_content > 0:
                        data, chunk_pos = read_binary_spec(
                            data_binary_str, uint_spec, chunk_pos
                        )
                        block_length = data[0]
                        (data_length,), chunk_pos = read_binary_spec(
                            data_binary_str, uint_spec_alt, chunk_pos
                        )

                        chunk_dict[external_id_str][block_idx] = zlib.decompress(
                            data_binary_str[chunk_pos : chunk_pos + data_length]
                        )

                        chunk_pos += data_length

                elif bd_id == BlockMarker.BLOCK_DATA_END:
                    pass
                elif (
                    bd_id == BlockMarker.BLOCK_STATUS
                    or bd_id == BlockMarker.BLOCK_CHECK_SUM
                ):
                    # Layout:
                    #   u32 v12      (=12)
                    #   u32 count    (= number of blocks)
                    #   u32 width    (=4, byte width of each entry)
                    #   count × u32  (status flags or checksums)
                    data, chunk_pos = read_binary_spec(
                        data_binary_str, block_test_spec, chunk_pos
                    )
                    count = data[1]
                    chunk_pos += 4  # skip _width field
                    entries_bytes = data_binary_str[chunk_pos : chunk_pos + count * 4]
                    entries = list(struct.unpack(f">{count}I", entries_bytes))
                    key = (
                        "statuses" if bd_id == BlockMarker.BLOCK_STATUS else "checksums"
                    )
                    block_metadata.setdefault(external_id_str, {})[key] = entries
                    chunk_pos += count * 4
                else:
                    num_skipped_dict[external_id_str] += 1
                    unknown_bd_ids[bytes(bd_id)] = (
                        unknown_bd_ids.get(bytes(bd_id), 0) + 1
                    )
                    chunk_pos = data_size

            pos += data_size

        elif chunk_type == ChunkMagic.SQLITE:
            break
        else:
            raise Exception("Invalid chunk type")

    if unknown_bd_ids:
        for bd_id, count in unknown_bd_ids.items():
            # Long bd_ids almost certainly mean we mis-aligned and grabbed a
            # blob of bytes thinking it was a name. Truncate the log so it
            # stays useful instead of dumping kilobytes per occurrence.
            preview = bd_id[:64]
            try:
                decoded = preview.decode("utf-16-be")
            except UnicodeDecodeError:
                decoded = preview.hex()
            suffix = "..." if len(bd_id) > 64 else ""
            logger.debug(
                f"Unknown bd_id encountered {count}×, len={len(bd_id)}B: "
                f"{decoded!r}{suffix}"
            )

    chunk_dict = dict(sorted(chunk_dict.items()))

    return chunk_dict, num_skipped_dict, file_header, block_metadata
