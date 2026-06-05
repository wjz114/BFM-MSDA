# BFM-MSDA: BFM-guided Multi-Source Domain Adaptation with Cauchy–Schwarz (CS) alignment for Cross-Subject EEG Decoding

Implementation of BFM-MSDA. The method combines foundation model guided source selection with feature-level and decision-level alignment via **Cauchy–Schwarz (CS) divergence** and **Conditional CS (CCS) divergence**.
[abstract]!(https://github.com/wjz114/BFM-MSDA/graphical_Abstract.png)

## Environment Setup

```bash
conda create -n bfm-msda python=3.9
conda activate bfm-msda
pip install -r requirements.txt
```

## Data Preparation

### BCI Competition IV 2a

Download the dataset and place the `.gdf` / `.mat` files under:

```
dataset/BCICIV_2a_gdf/
```

### Cho2017 (GigaDB subset)

Cho2017 is loaded automatically via [MOABB](https://neurotechx.github.io/moabb/) / Braindecode on first use. No manual download is required if MOABB is configured correctly.

## Source Selection Matrices

We have uploaded the example pairwise distances output by LaBraM ([Jiang](https://github.com/935963004/LaBraM)) in /distances.

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

Hope this code can be useful. I would appreciate you citing us (To be updated upon publication).

@misc{wu2026brainfoundationmodelmeets,
      title={When Brain Foundation Model Meets Cauchy-Schwarz Divergence: A New Framework for Cross-Subject Motor Imagery Decoding}, 
      author={Jinzhou Wu and Baoping Tang and Qikang Li and Yi Wang and Cheng Li and Shujian Yu},
      year={2026},
      eprint={2507.21037},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.21037}, 
}

