from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage

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


@pytest.fixture(scope="session")
def test001_clip_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test001.clip"


@pytest.fixture(scope="session")
def test001_png_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test001.png"


@pytest.fixture(scope="session")
def test001_clip(test001_clip_path: Path) -> ClipImage:
    return ClipImage.open(str(test001_clip_path))


@pytest.fixture(scope="session")
def test001_ground_truth(test001_png_path: Path) -> np.ndarray:
    return np.array(Image.open(test001_png_path))


@pytest.fixture(scope="session")
def test001_psd_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test001.psd"


@pytest.fixture(scope="session")
def test001_psd(test001_psd_path: Path) -> PSDImage:
    return PSDImage.open(str(test001_psd_path))


@pytest.fixture(scope="session")
def test002_clip_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test002.clip"


@pytest.fixture(scope="session")
def test002_psd_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test002.psd"


@pytest.fixture(scope="session")
def test002_clip(test002_clip_path: Path) -> ClipImage:
    return ClipImage.open(str(test002_clip_path))


@pytest.fixture(scope="session")
def test002_psd(test002_psd_path: Path) -> PSDImage:
    return PSDImage.open(str(test002_psd_path))


@pytest.fixture(scope="session")
def test003_clip_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test003.clip"


@pytest.fixture(scope="session")
def test003_psd_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test003.psd"


@pytest.fixture(scope="session")
def test003_clip(test003_clip_path: Path) -> ClipImage:
    return ClipImage.open(str(test003_clip_path))


@pytest.fixture(scope="session")
def test003_psd(test003_psd_path: Path) -> PSDImage:
    return PSDImage.open(str(test003_psd_path))
