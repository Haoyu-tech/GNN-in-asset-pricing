# Reproduction guide

## Current blockers

The [README](../README.md) describes the exact checkout status. `main.py` imports a missing `analysis.py`, and its model selector does not expose `CA0GNN`. Do not interpret installing dependencies as making the entire experiment runnable.

## Preparation

Install [requirements.txt](../requirements.txt). Set `AUTOENCODER_DATA_DIR` to a local folder containing `datashare.zip`, `month_ret.pkl`, `ff5.csv` and `UMD.csv`, then run [data_prepare_gnn.py](../data_prepare_gnn.py) from the root. Inspect input schema and SIC availability before generating graph panels.

Generated data belongs under `data/`; the source reads this directory even though some pickle artifacts were committed at the root. Neural inference currently requires a CUDA-capable PyTorch environment.

## Completion criteria

Restore the original analysis functions, provide an explicit GNN experiment entry point, and compare a baseline and GNN on identical train/validation/test periods. Record seeds, CUDA/PyTorch versions, graph construction and evaluation definitions. No end-to-end training or improvement claim was validated during this cleanup.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
