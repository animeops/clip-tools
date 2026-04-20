from typing import List, Tuple

import numpy as np

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
