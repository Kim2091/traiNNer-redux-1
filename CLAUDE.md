# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

traiNNer-redux is a PyTorch deep learning training framework for image super-resolution and restoration. Python 3.11+, PyTorch 2.10+.

## Common Commands

```bash
# Activate venv (always do this before running python)
source venv/Scripts/activate    # Windows (Git Bash)

# Install
pip install -e .[dev]           # Dev tools (ruff, pyright, pytest)
pip install -e .[dev,onnx]      # Dev + ONNX export support

# Lint & type check
ruff format --check
ruff check
pyright

# Tests
pytest ./tests                  # All tests
pytest ./tests/test_archs       # Single test directory

# Training & inference
python train.py -opt options/train/<ARCH>/<config>.yml
python test.py -opt options/test/<ARCH>/<config>.yml

# ONNX export
python convert_to_onnx.py -opt options/onnx/<ARCH>/<config>.yml
```

## Architecture

### Registry System

The codebase uses a registry/factory pattern. Components auto-register via decorators and are auto-discovered by filename convention:

| Registry | Decorator | File pattern | Location |
|---|---|---|---|
| `ARCH_REGISTRY` | `@ARCH_REGISTRY.register()` | `*_arch.py` | `traiNNer/archs/` |
| `SPANDREL_REGISTRY` | `@SPANDREL_REGISTRY.register()` | `*_arch.py` | `traiNNer/archs/` |
| `LOSS_REGISTRY` | `@LOSS_REGISTRY.register()` | `*_loss.py` | `traiNNer/losses/` |
| `DATASET_REGISTRY` | `@DATASET_REGISTRY.register()` | `*_dataset.py` | `traiNNer/data/` |
| `MODEL_REGISTRY` | `@MODEL_REGISTRY.register()` | — | `traiNNer/models/` |

Architectures registered with `SPANDREL_REGISTRY` are weight-compatible with the [Spandrel](https://github.com/chaiNNer-org/spandrel) ecosystem. `ARCH_REGISTRY` is for variants not in Spandrel.

### Config System

YAML configs in `options/` are parsed into typed `msgspec.Struct` classes defined in `traiNNer/utils/redux_options.py`. The `type` field in each config section selects the registered component by its function/class name (lowercase). Example: `type: span_f64` resolves to the `span_f64` function in `span_arch.py`.

### Training Flow

`train.py` -> `Config.load_config_from_file()` -> `build_dataset()` / `build_network()` / `build_model()` -> training loop with `model.optimize_parameters()`.

Model selection (`traiNNer/models/__init__.py`) is deterministic based on config flags: `SRModel` (default), `RealESRGANModel` (high-order degradation), `AEModel` (autoencoder).

### Arch Constraints (`traiNNer/archs/arch_info.py`)

This file tracks per-architecture constraints: FP16 incompatibility, channels-last performance, minimum HW size requirements, and temporal (video) architectures. Update it when adding architectures with special requirements.
