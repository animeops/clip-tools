# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

clip-tools is a Python library for parsing CLIP Studio Paint files (`.clip` and `.psd`) and exporting structured layers for anime production workflows. It handles binary format parsing, layer compositing, and organizes exports by production type (genga, layout, background, camera, etc.).

## Commands

```bash
# Install dependencies
uv sync

# Run tests with coverage
uv run pytest --cov=clip_tools

# Run a single test
uv run pytest tests/layer_test.py::TestClipFile::test_name -v

# Run the Gradio web interface
uv run python app/app.py

# CLI tool for processing files
uv run python scripts/playground.py <file_path> [--find-psds]
```

## Architecture

Three core classes form the main pipeline:

- **ClipImage** (`clip_tools/api/clip_image.py`) — Parses `.clip` binary files by extracting an embedded SQLite database and compressed image chunks. Entry point for file reading.
- **ClipLayer** (`clip_tools/api/clip_layer.py`) — Tree structure representing layer hierarchy. Handles compositing child layers into images, with caching. Supports text layers, resizable images, and vector data.
- **FileProcessor** (`clip_tools/file_processor.py`) — High-level export logic. Classifies folders by Japanese naming conventions (LO/Layout, G/Genga, BG, CAM, Paper) and exports to structured directories (`frames/`, `bg/`, `cam/`, `backing_paper/`, `meta/`).

Binary format parsing lives in `clip_tools/structs/` — chunk parsing, layer block decompression, vector data, text/image attributes. The CLIP file format is documented in `clip_tools/clip.md`.

## Key Details

- Python 3.9+ (`.python-version` pinned to 3.10.14)
- Uses `uv` for dependency management
- Folder type detection relies on Japanese naming conventions with fullwidth↔halfwidth normalization (`utils.py`)
- Layer compositing uses alpha blending and homography transforms via OpenCV and Pillow
- The package exports `FileProcessor`, `ClipImage`, `ClipLayer` from `clip_tools/__init__.py`
