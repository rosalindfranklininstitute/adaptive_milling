# Adaptive Milling

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/rosalindfranklininstitute/adaptive_milling/actions/workflows/python-test.yml/badge.svg)](https://github.com/rosalindfranklininstitute/adaptive_milling/actions/workflows/python-test.yml)

Adaptive milling plugins for [fibsemOS](https://github.com/fibsem-os/fibsem-os). This package currently contains:

- **AdaptivePolishing**: a milling strategy that uses machine learning to determine when polishing is complete.
- **AsymmetricFiducial**: a milling pattern that adds some asymmetry to the default Fiducial pattern.

## Installation

These instructions will install the latest version from the main branch by default. A release version can be installed by specifying a tag, for example `git+https://github.com/rosalindfranklininstitute/adaptive_milling.git@vX.X.X` would install tag `vX.X.X`. It is recommended to install the [latest release](https://github.com/rosalindfranklininstitute/adaptive_milling/releases/latest).

### [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

First, open a terminal in the directory you want your new environment to be created. In this example the environment is called `adaptive_milling_env`, which can be changed as required.

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
uv venv --python 3.11 adaptive_milling_env

# Activate the virtual environment (Windows)
adaptive_milling_env\Scripts\activate

# Install including AutoLamella from fibsemOS with the appropriate PyTorch backend for your machine
uv pip install git+https://github.com/rosalindfranklininstitute/adaptive_milling.git --extra ui --torch-backend auto
```

To activate the virtual environment on Linux systems: `source adaptive_milling_env/bin/activate`

### [miniforge](https://conda-forge.org/download/)

In this example a Conda environment will be created called `adaptive_milling`, which can be changed a required.

```Shell
# Create a virtual environment (Python 3.11 to match AutoScript's environment)
conda create -n adaptive_milling python=3.11

# Activate the virtual environment
conda activate adaptive_milling

# Install PyTorch (select appropriate compute platform, see table below)
python -m pip install pytorch --index-url <Index URL>

# Install including AutoLamella from fibsemOS
python -m pip install -e git+https://github.com/rosalindfranklininstitute/adaptive_milling.git[ui]
```

If using CUDA, the version must be below or equal to the the system CUDA version, which can be checked with the command `nvidia-smi`.

| Compute Platform | Index URL                              |
| ---------------- | -------------------------------------- |
| CPU              | https://download.pytorch.org/whl/cpu   |
| CUDA 11.3        | https://download.pytorch.org/whl/cu113 |
| CUDA 11.8        | https://download.pytorch.org/whl/cu118 |
| CUDA 12.6        | https://download.pytorch.org/whl/cu126 |
| CUDA 12.8        | https://download.pytorch.org/whl/cu128 |

## Usage

### Protocols

We recommend starting with [this protocol](src/example_configs/protocol-on-grid-ap.yaml) and customising to your requirements. You will need to set the `model_path` within the `AdaptivePolishing` config, either directly in the file or via the fibsemOS AutoLamella GUI.

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
