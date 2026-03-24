# Adaptive Milling

Adaptive milling plugins for [fibsemOS](https://github.com/fibsem-os/fibsem-os). This package currently contains:

* AdaptivePolishing milling strategy, which uses machine learning to determine when polishing is complete.
* AsymmetricFiducial milling pattern, which adds some asymmetry to the default Fiducial pattern to hopefully improve alignment.

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/rosalindfranklininstitute/adaptive_polish/actions/workflows/python-test.yml/badge.svg)](https://github.com/rosalindfranklininstitute/adaptive_polish/actions/workflows/python-test.yml)

## Installation

You can install the package by running the following commands (by specifying `ui`, AutoLamella will also be installed):

```
pip install -e .[ui]
```

or

```
uv sync --extra ui
```

If you are installing using `uv`, then you can specify the PyTorch platform by adding `--extra` followed by the appropriate extra name (version must be equal to or below the system CUDA version, which can be checked with the command `nvidia-smi`):

| Version   | Extra |
| --------- | ----- |
| CPU       | cpu   |
| CUDA 11.3 | cu113 |
| CUDA 11.8 | cu118 |
| CUDA 12.6 | cu126 |
| CUDA 12.8 | cu128 |

## Testing

You can run the package tests by running the following commands:

```
pip install .[test]
pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/adaptive_polish/issues) to submit bugs or request features.

## Contributions

If you would like to help contribute to profet, please read our [contribution](CONTRIBUTING.md) guide and [code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright Rosalind Franklin Institute, 2024.

Distributed under the terms of the Apache-2.0 license, adaptive polish is free and open source software.
