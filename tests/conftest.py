from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clip_tools import ClipImage


TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def test000_clip_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test000.clip"


@pytest.fixture(scope="session")
def test000_png_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test000.png"


@pytest.fixture(scope="session")
def test000_clip(test000_clip_path: Path) -> ClipImage:
    return ClipImage.open(str(test000_clip_path))


@pytest.fixture(scope="session")
def test000_ground_truth(test000_png_path: Path) -> np.ndarray:
    return np.array(Image.open(test000_png_path))
