# clip-tools

A Python library for parsing and exporting layers from CLIP Studio Paint files (`.clip` and `.psd`), built for anime production workflows.

## Features

- Parse CLIP Studio Paint binary file format and PSD files
- Extract and composite layer hierarchies
- Export organized layers by production type (genga, layout, background, camera, etc.)
- Gradio web interface for interactive use

## Installation

```bash
uv add clip-tools
```

Or for development:

```bash
git clone https://github.com/AnimeOps/clip-tools-anime.git
cd clip-tools-anime
uv sync
```

## Usage

### As a library

```python
from clip_tools import ClipImage, FileProcessor

# Parse a CLIP Studio Paint file
image = ClipImage("path/to/file.clip")
layers = image.layers

# Export structured layers
processor = FileProcessor("path/to/file.clip")
processor.export("output/directory")
```

### Web interface

```bash
uv run python app/app.py
```

### CLI

```bash
uv run python scripts/playground.py path/to/file.clip
```

## Requirements

- Python 3.9+

## License

MIT
