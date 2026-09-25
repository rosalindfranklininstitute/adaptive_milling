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
3. Create a desktop shortcut for AutoLamella (fibsemOS GUI). With the virtual environment activated, run the appropriate command for you operating system:

   ```cmd
   :: Windows command prompt: creates AutoLamella.bat
   echo @echo off > AutoLamella.bat & where fibsem-autolamella-ui >> AutoLamella.bat
   ```

   ```bash
   # Linux / macOS terminal: creates AutoLamella.sh
   printf '#!/bin/bash\n%s' $(which fibsem-autolamella-ui) > AutoLamella.sh
   chmod +x AutoLamella.sh
   ```

   Then create a shortcut to the script and place it on your desktop.

---

### Install using [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
uv venv --python 3.11

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Install including AutoLamella from fibsemOS with the appropriate PyTorch backend for your machine
uv pip install adaptive-milling[ui] --torch-backend auto
```

To activate the virtual environment on Linux systems: `source .venv/bin/activate`

---

### Install using [miniforge](https://conda-forge.org/download/)

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
conda create -p ./.venv python=3.11 pip

# Activate the virtual environment
conda activate ./.venv

# Install PyTorch (select appropriate compute platform, see table below)
python -m pip install torch torchvision --index-url <Index URL>

# Install including AutoLamella from fibsemOS
python -m pip install adaptive-milling[ui]
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

## Segmentation models

Models are available from zenodo: https://zenodo.org/records/21804785. We currently recommend using AM_SEM_All_V01.pth ([download link](https://zenodo.org/records/21804785/files/AM_SEM_All_V01.pth)).

If you wish to train your own segmentation models that are compatible with Adaptive Milling, or refine existing ones with new data, please use the package [AM-model-training](https://github.com/rosalindfranklininstitute/AM-model-training), also [available on PyPI](https://pypi.org/p/AM-model-training/). We encourage you to contribute your models and training data back to the project so that others may share the benefit. To do this, please contact [casper.berger@rfi.ac.uk](mailto:casper.berger@rfi.ac.uk).

## Testing

You can run the package tests using pytest:

```Shell
uv run pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/adaptive_milling/issues) to submit bugs or request features.

## Contributions

If you would like to help contribute to project, please read our [contribution](CONTRIBUTING.md) guide and [code of conduct](CODE_OF_CONDUCT.md).

If you would like to contribute data towards training new segmentation models, please contact [casper.berger@rfi.ac.uk](mailto:casper.berger@rfi.ac.uk).

## License

Copyright Rosalind Franklin Institute, 2024.

Distributed under the terms of the Apache-2.0 license with "Commons Clause" License Condition v1.0, see the [license](LICENSE) for further details.
