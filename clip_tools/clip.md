# CLIP Studio Paint File Explanation

## Top-level structure

A `.clip` file is a concatenation of two regions:

```
[binary chunk region][SQLite3 database]
```

The split point is found by scanning for the ASCII string `SQLite format 3`.
The SQLite database holds all relational metadata (layers, canvas, mipmaps,
brushes, vector references, etc.). The binary region holds the compressed
pixel/vector data referenced by external-id strings from the SQLite side.

See `clip_tools/io.py::split_clip_binary` and `load_sqlite`.

## How pixel data is referenced

The binary region is parsed into a dictionary keyed by external-id strings:

```
clip_data : Dict[str, Union[Dict[int, bytes], np.ndarray]]
```

- `Dict[int, bytes]` values = raster block chunks (`block_idx -> zlib-compressed pixel block`).
- `np.ndarray` values = vector data already rasterized into a full-canvas RGBA array.

Each external-id ties back to a specific `(table, column, row)` via the
`ExternalTableAndColumnName` SQLite table. `clip_tools/processing.py::build_external_id_map`
builds this mapping.

**Gotchas observed:**
- `ExternalTableAndColumnName` can reference columns that *don't exist* in the
  target table (schema versioning). Parsers must skip missing columns.
- `clip_data` often contains empty `{}` dict entries — an external-id was
  mentioned in the binary but no actual chunks followed. These silently fail
  the classifier's `isinstance(value, dict) and value` check and are dropped.
- Only external-ids actually *written into* the binary chunk stream appear
  in `clip_data`; many Offscreens declared in SQLite never show up here.

## Offscreen table: per-layer mipmap pyramid + thumbnail

Each layer (raster or group) has **multiple** `Offscreen` rows — typically:

```
5 mipmap levels         ThisScale = 100.0, 50.0, 25.0, 12.5, 6.25   (in MipmapInfo)
1 extra thumbnail       not in MipmapInfo, ~512×512 preview cache
```

(Some files have a 6th level at `ThisScale = 3.125`.)

The `MipmapInfo` table indexes only the mipmap pyramid. The extra "NO_MIP"
Offscreen is what CLIP Studio uses as a thumbnail cache. Critically:

- **For group layers, the full-resolution top mipmap often has no chunk data
  stored in the binary at all** — only the 512×512 thumbnail does. This means
  a group's cached 512×512 preview is frequently the *only* rendered copy
  available from the file.
- **For leaf rasters inside caching groups** (e.g. `LO/A/1/line` structures),
  *none* of the Offscreens may have chunk data. CLIP flattens the strokes into
  the parent group's cache on save.

This has a concrete implication for compositors: a group with no leaf-raster
children can still have renderable content (its cached thumbnail), and walking
children alone produces a blank buffer. See `clip_tools/api/clip_layer.py`
`composite()` for the fallback path.

## Offscreen.Attribute blob

Each `Offscreen` row in the SQLite database has an `Attribute` column — a
binary blob describing the raster layer's geometry, initial color, and per-block
compressed sizes.

The blob is a **self-describing TLV-like structure** with a 16-byte section-size
table up front, followed by three sections (`Parameter`, `InitColor`, `BlockSize`)
separated by `9`-valued boundary markers. Section names are UTF-16BE encoded.

### Layout

```
offset  size  field                    notes
------  ----  -----------------------  -----
  0     16    section_sizes (4 x u32)  see below
  16     4    boundary                 always 9
  20    18    "Parameter" (utf-16be)
  38     8    width, height
  46     8    cols, rows               block grid; nblocks = cols * rows
  54    16    color_mode params        (33, 1, num_channels, 5) — inferred labels
  70    16    block_geom               (65536, 4, 1024, 1) — 256×256 blocks, 32×32 subblocks
  86    16    block_dims               (block_w, 65536, block_h, block_stride)
 102    16    subblock_dims            (8, 8, 0, 0) — inferred
 118     4    boundary                 always 9
 122    18    "InitColor" (utf-16be)
 140     4    initcolor_magic          always 20
 144    16    init_color               (has_color, packed_rgba, nchan, nchan)
(160)  (16)   init_color_extra         only present when has_color==1; zeros in all samples
 ...    4    boundary                 always 9
 ...   18    "BlockSize" (utf-16be)
 ...   12    blocksize_hdr            (magic=12, nblocks, nchan)
 ...    *    block_sizes              nblocks × u32, per-block compressed byte size
```

### `section_sizes` (self-describing table)

```
section_sizes[0] = 16                   # size of this table itself
section_sizes[1] = 102                  # length of Parameter section (from B1 through end of block)
section_sizes[2] = 42  or  58           # length of InitColor section (42 base, +16 if has_color)
section_sizes[3] = 34 + 4 * nblocks     # length of BlockSize section
```

Any parser can skip entire sections using this table without knowing internal
layout.

### `init_color`

`(has_color: u32, packed_rgba: u32, nchan: u32, nchan: u32)`

- When `has_color == 0`: layer has no default fill. `packed_rgba` is ignored.
- When `has_color == 1`: `packed_rgba` is the RGBA fill color as a big-endian
  u32 (e.g. `0xFFFFFFFF` = opaque white). An additional 16 zero bytes follow.

The 用紙 (paper / canvas backing) layer is the one layer that reliably has
`has_color == 1`.

### `block_sizes`

`nblocks × u32` — the compressed byte size of each 256×256 block's data. The
block grid is row-major; `nblocks == cols * rows`.

Per-block sizes vary wildly based on content:
- Uniform-colored blocks (like a white paper fill) compress to ~104 bytes each.
- Content-varying blocks range from a few hundred bytes to ~90k.

## Open questions / unresolved

1. **`color_mode` family values** — `(33, 1, ?, 5)` constant across every
   sample analyzed. Labels (`color_mode`, `alpha_flag`, `bit_depth_enum`) are
   inferred from position. Samples with different color settings (grayscale,
   16-bit, CMYK) would be needed to confirm semantics.

2. **`initcolor_magic = 20`** — either a section-type marker or a body-length
   field. In the no-color case the body is exactly 20 bytes, so the two
   hypotheses are indistinguishable from current samples.

3. **`init_color_extra` (16 bytes)** — only present when `has_color == 1`,
   always zeros in observed samples. Likely holds data only for non-white /
   non-opaque init colors. A layer with a tinted paper color would crack it.

4. **`text_attributes.py`** — parser still has ~30 unnamed uint reads marked
   `"?"`. No decoding attempted; needs sample files containing text layers.

5. **`vector.py`** — stroke records have `mystery_0` through `mystery_6` fields.
   The `wn_04_009_LOSA.clip` sample has real vector strokes, so these are now
   investigable.

6. **`ExternalTableAndColumnName` schema drift** — the `wn_04_009_LOSA.clip`
   sample references a `BankData` column that doesn't exist in the target
   table for this file version. The parser currently skips missing columns
   with a debug log; the semantics of `BankData` (vs `BlockData` / `VectorData`)
   are unknown.

## Classification pipeline (`process_clip_data`)

Once `clip_data` is built, each entry is classified by matching its external-id
back to an `Offscreen` row and reading the layer's SQLite metadata:

```
raster_dict  : Dict[layer_id, LayerEntry]   # usable rendering entries
aux_list     : List[LayerEntry]             # everything else
```

Classification order:

1. **invalid** — Offscreen's `LayerId` not present in the `Layer` table.
2. **clipped** — layer's `LayerClip != 0` (clipping-mask layer).
3. **group** — `LayerFolder != 0`. Stored in **both** `raster_dict` and
   `aux_list`; the compositor prefers child recursion but falls back to the
   cached group raster when children produce an empty buffer.
4. **mipmap** — Offscreen is in `MipmapInfo` but not the top level
   (`ThisScale != 100`).
5. **other** — Offscreen is *not* in `MipmapInfo` at all (the NO_MIP thumbnail case).
6. **raster** — everything else. Leaf raster layer at full resolution.
7. **vector** — value is an `np.ndarray` (already rasterized in `process_chunk_binary`).

## Reverse-engineering technique notes

The Offscreen.Attribute decoding was cracked by:

1. Dumping the same structure from 4+ layers in parallel and diffing.
2. Size-delta analysis: the paper layer was 16 bytes longer than others,
   which pinpointed the has_color conditional payload.
3. Packed-integer recognition: `0xFFFFFFFF` on a layer named 用紙 right
   after an "InitColor" marker was hard to miss.
4. Offset arithmetic: `section_sizes[1] = 102` equals the byte distance
   from end-of-UNK1 to the B2 boundary — revealed the whole table's purpose.
5. Cross-mipmap validation: checking that `num_blocks == cols × rows` held
   across 5 resolution levels of the same layer (not just one) is much stronger
   evidence than a single coincidence.
