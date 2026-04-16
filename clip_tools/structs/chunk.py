from typing import Dict, Any, Tuple
import struct
import zlib
import pandas as pd
import logging

from clip_tools.utils import read_binary_spec
from clip_tools.constants import DEBUG
from .vector import process_vector_binary

logger = logging.getLogger(__name__)


def process_chunk_binary(
    clip_binary_str: bytes,
    canvas_shape: Tuple[int, int],
    external_id_map: Dict[str, Dict[str, Any]],
    brush_styles: pd.DataFrame,
) -> Tuple[Dict[str, Dict[int, bytes]], Dict[str, int]]:
    CSF_CHUNK = b"CSFCHUNK"
    CHUNK_HEADER = b"CHNKHead"
    CHUNK_EXTERNAL = b"CHNKExta"
    CHUNK_SQLITE = b"CHNKSQLi"
    CHUNK_FOOTER = b"CHNKFoot"

    BLOCK_DATA_BEGIN_CHUNK = "BlockDataBeginChunk".encode("utf-16be")
    BLOCK_DATA_END_CHUNK = "BlockDataEndChunk".encode("utf-16be")
    BLOCK_STATUS = "BlockStatus".encode("utf-16be")
    BLOCK_CHECK_SUM = "BlockCheckSum".encode("utf-16be")

    clip_header_spec = struct.Struct(">8sQQ")
    chunk_header_spec = struct.Struct(">8sQ")
    external_header_spec = struct.Struct(">Q40sQ")
    block_test_spec = struct.Struct(">II")
    uint_spec = struct.Struct(">I")
    uint_spec_alt = struct.Struct("<I")  # Little endian
    block_header_spec = struct.Struct(">I12xI")

    zlib_delimiter = b"\x78"

    chunk_dict = {}
    num_skipped_dict = {}

    num_external_ids = 0

    pos = 0
    data, pos = read_binary_spec(clip_binary_str, clip_header_spec, pos)
    _, filesize, _ = data

    while pos < filesize:
        data, pos = read_binary_spec(clip_binary_str, chunk_header_spec, pos)
        chunk_type, chunk_length = data
        # print(chunk_type)

        if chunk_type == CHUNK_HEADER:
            # Skip the header for now
            pos += chunk_length
        elif chunk_type == CHUNK_EXTERNAL:
            data, pos = read_binary_spec(clip_binary_str, external_header_spec, pos)

            _, external_id, data_size = data
            external_id_str = external_id.decode("UTF-8")

            num_skipped_dict[external_id_str] = 0

            if external_id_str not in external_id_map:
                raise Exception(
                    f"External ID {external_id_str} not found in CLIP SQLite database"
                )
            else:
                external_id_map[external_id_str]["found"] = True

            chunk_dict[external_id_str] = {}
            data_binary_str = clip_binary_str[pos : pos + data_size]

            num_external_ids += 1

            logger.debug(
                f"Processing external ID #{num_external_ids} with ID {external_id_str}..."
            )

            chunk_pos = 0
            while chunk_pos < data_size:
                # Read block header
                data, chunk_pos = read_binary_spec(
                    data_binary_str, block_test_spec, chunk_pos
                )

                if data == (88, 72):
                    chunk_pos -= 8
                    chunk_dict[external_id_str] = process_vector_binary(
                        data_binary_str, data_size, canvas_shape, brush_styles
                    )
                    chunk_pos = data_size
                    continue

                if (
                    data[1]
                    == uint_spec.unpack(BLOCK_DATA_BEGIN_CHUNK[: uint_spec.size])[0]
                ):
                    # print("Case A")
                    str_length = data[0]
                    chunk_pos -= uint_spec.size
                else:
                    # print("Case B")
                    str_length = data[1]
                    data_length = data[0]

                # print(data)
                bd_id = data_binary_str[chunk_pos : chunk_pos + (str_length * 2)]
                # print("bd_id: ", bd_id.hex())
                chunk_pos += str_length * 2

                if bd_id == BLOCK_DATA_BEGIN_CHUNK:
                    # print("BLOCK_DATA_BEGIN_CHUNK")
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

                elif bd_id == BLOCK_DATA_END_CHUNK:
                    pass
                    # print("BLOCK_DATA_END_CHUNK")
                elif bd_id == BLOCK_STATUS:
                    # print("BLOCK_STATUS")
                    data, chunk_pos = read_binary_spec(
                        data_binary_str, block_test_spec, chunk_pos
                    )
                    str_length = data[1]
                    chunk_pos += str_length * 4
                    # chunk_pos += 28
                    # chunk_pos += 12
                    # import pdb; pdb.set_trace()
                elif bd_id == BLOCK_CHECK_SUM:
                    # print("BLOCK_CHECK_SUM")
                    data, chunk_pos = read_binary_spec(
                        data_binary_str, block_test_spec, chunk_pos
                    )
                    str_length = data[1] + 1
                    chunk_pos += str_length * 4
                else:
                    # import pdb; pdb.set_trace()
                    # print("Unknown block... skipping")
                    num_skipped_dict[external_id_str] += 1
                    chunk_pos = data_size
                    pass

            pos += data_size

        elif chunk_type == CHUNK_SQLITE:
            break
        else:
            # print(chunk_type)
            raise Exception("Invalid chunk type")

    chunk_dict = dict(sorted(chunk_dict.items()))

    # Sort dict before returning
    return chunk_dict, num_skipped_dict
