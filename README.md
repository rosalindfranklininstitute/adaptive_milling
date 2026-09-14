# Adaptive Milling

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/rosalindfranklininstitute/adaptive_milling/actions/workflows/python-test.yml/badge.svg)](https://github.com/rosalindfranklininstitute/adaptive_milling/actions/workflows/python-test.yml)

Adaptive milling plugins for [fibsemOS](https://github.com/fibsem-os/fibsem-os). This package currently contains:

- **AdaptivePolishing**: a milling strategy that uses machine learning to determine when polishing is complete.
- **AsymmetricFiducial**: a milling pattern that adds some asymmetry to the default Fiducial pattern.

## Installation

Adaptive Milling is [available from PyPI](https://pypi.org/p/adaptive-milling) and can be installed via the following steps:

1. Decide where you want to create your Python virtual environment and open a terminal in that location.
2. Follow the one of the sets of instructions below. We recommend using 'uv' as it handles installing the appropriate torch backend automatically.
3. Create a desktop shortcut for AutoLamella (fibsemOS GUI):
  - Run `fibsem-autolamella-ui` from the virtual environment created during step 3.
  - Tools → Create Desktop Shortcut...

In order to get AutoLamella set up for your system, please see the ['Getting started' documentation](https://www.fibsemos.org/docs/getting-started/) for fibsemOS.

---

### Install using [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
uv venv --python 3.11

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Install including AutoLamella from fibsemOS with the appropriate PyTorch backend for your machine
uv pip install adaptive-milling --extra ui --torch-backend auto
```

To activate the virtual environment on Linux systems: `source .venv/bin/activate`

---

### Install using [miniforge](https://conda-forge.org/download/)

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
conda create -p ./.venv python=3.11

# Activate the virtual environment
conda activate ./.venv

# Install PyTorch (select appropriate compute platform, see table below)
python -m pip install torch torchvision --index-url <Index URL>

# Install including AutoLamella from fibsemOS
python -m pip install -e adaptive-milling[ui]
```

If using CUDA, the version must be below or equal to the the system CUDA version, which can be checked with the command `nvidia-smi`.

| Compute Platform | Index URL                              |
| ---------------- | -------------------------------------- |
| CPU              | https://download.pytorch.org/whl/cpu   |
| CUDA 11.3        | https://download.pytorch.org/whl/cu113 |
| CUDA 11.8        | https://download.pytorch.org/whl/cu118 |
| CUDA 12.6        | https://download.pytorch.org/whl/cu126 |
| CUDA 12.8        | https://download.pytorch.org/whl/cu128 |

---


## Usage

Please see the [user guide](USERGUIDE.md).

## Testing

You can run the package tests using pytest:

```Shell
uv run pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/adaptive_milling/issues) to submit bugs or request features.

## Contributions

If you would like to help contribute to project, please read our [contribution](CONTRIBUTING.md) guide and [code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright Rosalind Franklin Institute, 2024.

Distributed under the terms of the Apache-2.0 license with "Commons Clause" License Condition v1.0, see the [license](LICENSE) for further details.
