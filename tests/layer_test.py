from typing import List, Tuple

import numpy as np

from psd_tools import PSDImage

from clip_tools import ClipImage, ClipLayer


def _walk(layers, depth: int = 0) -> List[Tuple[int, ClipLayer]]:
    out: List[Tuple[int, ClipLayer]] = []
    for layer in layers:
        out.append((depth, layer))
        if layer.is_group():
            out.extend(_walk(layer, depth + 1))
    return out


class TestTest000:
    def test_canvas_size(self, test000_clip: ClipImage):
        assert test000_clip._canvas_size == (700, 1400)

    def test_top_level_layer_count(self, test000_clip: ClipImage):
        assert len(test000_clip) == 3

    def test_total_layer_count(self, test000_clip: ClipImage):
        assert len(_walk(test000_clip)) == 3

    def test_layer_names(self, test000_clip: ClipImage):
        assert [l.name for l in test000_clip] == ["用紙", "layer1", "layer2"]

    def test_layer_ids(self, test000_clip: ClipImage):
        assert [int(l.layer_id) for l in test000_clip] == [4, 3, 5]

    def test_layer_types(self, test000_clip: ClipImage):
        assert [l.layer_type for l in test000_clip] == ["raster"] * 3

    def test_no_groups(self, test000_clip: ClipImage):
        assert all(not l.is_group() for l in test000_clip)

    def test_all_visible(self, test000_clip: ClipImage):
        assert all(bool(l.visible) for l in test000_clip)

    def test_bboxes(self, test000_clip: ClipImage):
        assert [tuple(int(x) for x in l.bbox) for l in test000_clip] == [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ]

    def test_composite_matches_golden(
        self, test000_clip: ClipImage, test000_ground_truth: np.ndarray
    ):
        composited = np.array(test000_clip.composite())
        assert composited.shape == test000_ground_truth.shape
        assert np.array_equal(composited, test000_ground_truth)


class TestTest001:
    def test_canvas_size(self, test001_clip: ClipImage):
        assert test001_clip._canvas_size == (1200, 1600)

    def test_top_level_layer_count(self, test001_clip: ClipImage):
        assert len(test001_clip) == 3

    def test_total_layer_count(self, test001_clip: ClipImage):
        assert len(_walk(test001_clip)) == 3

    def test_layer_names(self, test001_clip: ClipImage):
        assert [l.name for l in test001_clip] == ["Paper", "Layer 1", "Layer 2"]

    def test_layer_ids(self, test001_clip: ClipImage):
        assert [int(l.layer_id) for l in test001_clip] == [4, 3, 5]

    def test_layer_types(self, test001_clip: ClipImage):
        assert [l.layer_type for l in test001_clip] == ["raster", "raster", "vector"]

    def test_has_vector_layer(self, test001_clip: ClipImage):
        assert any(l.layer_type == "vector" for l in test001_clip)

    def test_no_groups(self, test001_clip: ClipImage):
        assert all(not l.is_group() for l in test001_clip)

    def test_all_visible(self, test001_clip: ClipImage):
        assert all(bool(l.visible) for l in test001_clip)

    def test_composite_close_to_psd_golden(
        self, test001_clip: ClipImage, test001_ground_truth: np.ndarray
    ):
        # test001.png is the PSD's composite — i.e. what CLIP Studio itself
        # renders. Our pipeline's vector rasterizer is known to be incomplete
        # (single-pixel Bresenham, no texture stamping), so byte-equality isn't
        # expected. This test checks that we don't drift *further* from CLIP
        # than our current baseline.
        composited = np.array(test001_clip.composite())

        # Flatten both onto white to make the metric meaningful regardless of
        # alpha-channel encoding differences.
        def on_white(im: np.ndarray) -> np.ndarray:
            alpha = im[..., 3:4].astype(np.float32) / 255
            return (im[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)

        ours = on_white(composited)
        golden = on_white(test001_ground_truth)
        diff = np.abs(ours.astype(int) - golden.astype(int))

        # Current baseline: mean diff ~3.6, 6.3% pixels differ >10.
        # Bounds are loose; tighten as the vector rasterizer improves.
        assert diff.mean() < 10, f"mean pixel diff {diff.mean():.2f} exceeds 10"
        bad_pct = 100 * (diff.max(-1) > 10).sum() / (diff.shape[0] * diff.shape[1])
        assert bad_pct < 15, f"{bad_pct:.2f}% of pixels differ > 10 (baseline ~6.3%)"

    def test_psd_has_matching_layer_structure(
        self, test001_clip: ClipImage, test001_psd: PSDImage
    ):
        # The .psd was exported from CLIP Studio with layer structure preserved
        # and should have the same three top-level layers as the .clip file.
        psd_names = [l.name for l in test001_psd.descendants() if l.kind == "pixel"]
        clip_names = [l.name for l in test001_clip]
        assert psd_names == clip_names
