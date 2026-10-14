<div align="center">

# Graph-Aware Asset Pricing

Characteristic-based autoencoders, attention models and SIC industry graphs.

![Stack: PyTorch](https://img.shields.io/badge/Stack-PyTorch-EE4C2C?style=flat-square) ![Graph: SIC industry](https://img.shields.io/badge/Graph-SIC%20industry-0F766E?style=flat-square) ![Status: Incomplete pipeline](https://img.shields.io/badge/Status-Incomplete%20pipeline-B45309?style=flat-square)

[Quick start](#quick-start) · [Project map](#project-map) · [Reproduction guide](docs/REPRODUCIBILITY.md)

</div>

---

## Overview

Research code exploring whether industry relationships help represent firm characteristics and latent return factors. The repository contains baseline model definitions, attention variants and a SIC graph model.

## Implementation status

| Component | Current state |
|---|---|
| Baselines | FF, PCA, IPCA and autoencoder definitions are present |
| Attention models | CA0Att and CA0AttM are exposed in `main.py` |
| Graph model | CA0GNN is defined but not wired into the main selector |
| Full experiment | Blocked by missing `analysis.py` and external model inputs |
| Neural inference | The main driver explicitly uses CUDA |

Model code and saved artifacts alone do not establish a performance improvement.

## Quick start

Prepare the external inputs before running preprocessing:

```powershell
python -m pip install -r requirements.txt
$env:AUTOENCODER_DATA_DIR = "C:/data/Autoencoder_data"
python data_prepare_gnn.py
```

The input folder must contain `datashare.zip`, `month_ret.pkl`, `ff5.csv` and `UMD.csv` in the expected schemas. Preprocessing writes generated data into `data/`.

> **Training prerequisite:** restore the original `analysis.py`, prepare the model inputs, and use a compatible PyTorch/CUDA environment before running the experiment driver.

## Project map

| Path | Role |
|---|---|
| [data_prepare_gnn.py](data_prepare_gnn.py) | Prepare characteristics, returns and graph panels |
| [main.py](main.py) | Baseline and attention experiment driver |
| [models/](models/) | Baseline, autoencoder, attention and graph models |
| [utils.py](utils.py) | Shared characteristic list and settings |
| [R_squares/](R_squares/) | Saved evaluation summaries |
| [saved_models/](saved_models/) | Saved model artifacts |

## Evaluation roadmap

Restore the missing analysis functions, expose an explicit GNN experiment, and compare models on identical train/validation/test periods. Record graph construction, seeds, normalization and evaluation definitions.

## License

See [LICENSE](LICENSE) for the existing source license.

## Documentation

- [Reproduction guide](docs/REPRODUCIBILITY.md): inputs, execution order and validation limits.
- [Code and artifact map](docs/REPOSITORY_MAP.md): source and configuration paths.

<sub>Research artifacts reflect the recorded experiment settings. Full results were not regenerated during documentation cleanup.</sub>
