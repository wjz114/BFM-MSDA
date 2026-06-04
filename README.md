# BFM-MSDA: Multi-Source Domain Adaptation for Cross-Subject EEG Decoding

Official implementation of multi-source domain adaptation (MSDA) for motor imagery EEG classification across subjects. The method combines **EEGNet** with feature-level alignment via **Cauchy–Schwarz (CS) divergence** and decision-level alignment via **Conditional CS (CCS) divergence**, together with adaptive source selection from precomputed inter-subject distance matrices.

## Directory Layout

```
submission/
├── README.md
├── requirements.txt
├── configs/                  # Experiment YAML configs
├── model/                    # EEGNet teacher model
├── dataloader/               # BCI IV 2a and Cho2017 loaders
├── utils/                    # CS/CCS losses, training helpers
├── filters.py                # Bandpass filters (used by dataloader)
├── BFM-MSDA_main.py          # Main MSDA training
├── evaluation.py             # Checkpoint evaluation
├── baseline_subject_scaling.py
├── baseline_scaling_ordered.py
├── run_adaptive_selection.py
├── compare_selections_2a.py
├── source_selection.py       # Distance heatmaps and MDS plots
├── 2a_pairwise_distances.csv
├── cho_2017_pairwise_distances_data.csv
├── dataset/                  # Place raw EEG data here (not included)
├── checkpoints/              # Saved model weights (generated)
└── results/                  # Selection outputs and metrics (generated)
```

## Environment Setup

```bash
conda create -n bfm-msda python=3.9
conda activate bfm-msda
pip install -r requirements.txt
```

All commands below should be run from the `submission/` directory.

## Data Preparation

### BCI Competition IV 2a

Download the dataset and place the `.gdf` / `.mat` files under:

```
dataset/BCICIV_2a_gdf/
```

The path is configured in `configs/bcicompet2a_config.yaml`.

### Cho2017 (GigaDB subset)

Cho2017 is loaded automatically via [MOABB](https://neurotechx.github.io/moabb/) / Braindecode on first use. No manual download is required if MOABB is configured correctly.

## Reproducible Pipeline

### 1. Visualize inter-subject distances (optional)

Generate heatmaps and MDS plots from the included distance matrices:

```bash
# BCI IV 2a
python source_selection.py --dataset bciciv2a

# Cho2017
python source_selection.py --dataset cho2017
```

Outputs PNG files in the current directory (e.g. `pairwise_distances_heatmap_2a.png`, `MDS_2a.png`).

### 2. Adaptive source selection

Run GMM-based and softmax-temperature selection on the precomputed distance CSVs:

```bash
python run_adaptive_selection.py
```

Outputs are written to `results/`:

- `<dataset>__gmm2__pairs.csv`, `<dataset>__gmm2__summary.csv`
- `<dataset>__softmax_tau__pairs.csv`, `<dataset>__softmax_tau__summary.csv`
- Diagnostic plots under `results/plots/`

### 3. Train MSDA model

Main training with CS/CCS alignment and adaptive source weighting:

```bash
python BFM-MSDA_main.py --config_name bcicompet2a_config --gpu_num 0
```

Checkpoints are saved under `checkpoints/<task>_<model_name>/<model_type>/`.

For Cho2017, use `--config_name Cho_config`.

### 4. Compare selection strategies (BCI IV 2a)

Train and compare manual, GMM, and softmax-tau source selection:

```bash
python compare_selections_2a.py --epochs 200 --gpu_num 0
```

Requires adaptive selection outputs in `results/` (step 2). Results are saved to `results/selection_comparison_2a.csv`.

### 5. Baseline experiments (no domain adaptation)

Random source scaling:

```bash
python baseline_subject_scaling.py --epochs 300 --gpu_num 0
```

Ordered source addition (e.g. most similar first, for negative-transfer analysis):

```bash
python baseline_scaling_ordered.py --source_order distance:2a_pairwise_distances.csv --epochs 300
```

Other orderings: `ascending`, `descending`, or `distance_reverse:2a_pairwise_distances.csv`.

### 6. Evaluate trained checkpoints

```bash
python evaluation.py --config_name bcicompet2a_config --gpu_num 0
```

Loads checkpoints from `checkpoints/` and reports accuracy and Cohen's kappa.

## Key Hyperparameters

Default settings are in `configs/bcicompet2a_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 300 | Training epochs |
| `lr` | 0.001 | Adam learning rate |
| `alpha` | 0.4 | Feature-level CS loss weight |
| `beta` | 1.4 | Decision-level CCS loss weight |
| `warm_up` | 100 | Epochs before enabling alignment losses |
| `batch_size` | 32 | Mini-batch size |

## Citation

If you use this code, please cite the corresponding paper (details to be added upon publication).

## License

See the repository license file for terms of use.
