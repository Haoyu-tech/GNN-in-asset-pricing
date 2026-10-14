# Autoencoder Asset Pricing with SIC Graph Neural Networks

[Reproduction guide](docs/REPRODUCIBILITY.md) · [Code and artifact map](docs/REPOSITORY_MAP.md)

Research code for characteristic-based latent-factor asset pricing, with baseline autoencoders, attention variants, and a SIC graph model.

## Implementation status

The checkout includes model definitions and a data-preparation script, but the full experiment is **not reproducible from a fresh clone yet**:

- `main.py` imports `analysis.py`, which is absent. Restore the original analysis module before training through this entry point.
- `models/CA0GNN.py` defines `CA0GNN`, but `main.py` currently selects only FF, PCA, IPCA, CA0, CA0Att and CA0AttM. The GNN experiment needs a separate entry point or explicit integration.
- Neural-model inference in `main.py` explicitly uses CUDA.
- Raw research inputs and the generated `data/` panel are not included. Root-level pickle files do not replace the `data/` paths read by the models.

The presence of code or saved artifacts does not establish an improvement over the baseline. Compare models using the same data, training windows and evaluation code after restoring the missing pieces.

## Repository map

| Path | Purpose |
|---|---|
| [data_prepare_gnn.py](data_prepare_gnn.py) | Prepare characteristics, returns, factors and SIC graph data |
| [main.py](main.py) | Baseline and attention model experiment driver |
| [models/](models/) | FF, PCA, IPCA, autoencoder, attention and graph implementations |
| [utils.py](utils.py) | Characteristic list and shared settings |
| [Code_factor94/](Code_factor94/) | Additional characteristic data-loading code |
| [R_squares/](R_squares/) | Saved evaluation summaries |
| [saved_models/](saved_models/) | Committed model artifacts |

## Environment and data preparation

Run from the repository root. The dependency list is derived from the committed source and is not a validated version lock.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:AUTOENCODER_DATA_DIR = "C:/data/Autoencoder_data"
python data_prepare_gnn.py
```

The external data folder must contain `datashare.zip`, `month_ret.pkl`, `ff5.csv` and `UMD.csv` in the schemas expected by the preprocessing code. It writes generated panels into `data/`. Use trusted pickle inputs only.

After restoring `analysis.py` and preparing the model inputs, an explicit baseline selection avoids the default all-characteristic omission experiment:

```powershell
python main.py --Model PCA --K 3 --omit_char ""
```

This command is a reproduction target; it has not been executed against the external data. Check [the reproduction guide](docs/REPRODUCIBILITY.md) for blockers and validation limits.

## Research scope

The project studies whether industry relationships help represent firm characteristics and latent return factors. Other graph architectures, additional industry resolutions, and performance claims require implementation and a controlled comparison. Keep source attribution in the existing [LICENSE](LICENSE).
