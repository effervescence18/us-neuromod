# us-neuromod

ML models for transcranial focused ultrasound pressure field prediction
using the TFUScapes dataset.

## Project Overview

Predicting 3D acoustic pressure fields from skull pseudo-CT volumes and
transducer placement, using deep learning to replace expensive k-Wave
simulations.

## Dataset

- TFUScapes: https://huggingface.co/datasets/vinkle-srivastav/TFUScapes
- Paper: https://arxiv.org/abs/2505.12998
- Code: https://github.com/CAMMA-public/TFUScapes

The dataset is NOT included in this repo (too large). Download separately.

## Project Structure

- `notebooks/` — Exploration and analysis notebooks
- `src/` — Reusable Python modules (data loading, models, utils)
- `configs/` — Experiment configuration files
- `results/` — Figures, logs, metrics (mostly gitignored)

## Setup
```bash
pip install numpy matplotlib
```


# Forward Model: 3D U-Net for tFUS Pressure Prediction

Predicts acoustic pressure fields from CT skull images + transducer placement using a 3D U-Net, replacing expensive k-Wave simulations with fast neural network inference.

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision scipy numpy matplotlib datasets huggingface_hub
```

### 2. Download dataset (run once, ~200 GB for full dataset)
```bash
python forward_model_unet.py --download_only --data_dir /path/to/tfuscapes

# Or dev mode for quick testing (~50 GB):
python forward_model_unet.py --download_only --data_dir /path/to/tfuscapes --mode dev
```

### 3. Train
```bash
# Full dataset, default hyperparameters (base_features=16, ~6.5M params)
python forward_model_unet.py --data_dir /path/to/tfuscapes --mode full --epochs 100

# Larger model (base_features=32, ~25M params — recommended for full dataset)
python forward_model_unet.py --data_dir /path/to/tfuscapes --mode full --epochs 100 --base_features 32

# Resume from checkpoint
python forward_model_unet.py --data_dir /path/to/tfuscapes --resume runs/<run_dir>/checkpoints/best_model.pt --epochs 50
```

### 4. Outputs
Each run creates a timestamped directory under `runs/` containing:
```
runs/20250211_143000_full_bf32/
├── config.json          # exact hyperparameters for reproducibility
├── training.log         # full training log
├── history.json         # loss curves (train & val per epoch)
├── checkpoints/
│   ├── best_model.pt    # lowest validation loss
│   ├── checkpoint_epoch_10.pt
│   └── final_model.pt
└── visualizations/
    ├── predictions_epoch10.png
    └── predictions_epoch20.png
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--base_features` | 16 | Try 32 for full dataset (25M vs 6.5M params) |
| `--lr` | 1e-4 | Adam learning rate |
| `--weight_alpha` | 1.0 | Focal spot emphasis in loss (higher = more focus on peak) |
| `--target_size` | 128 | Downsample 256³ → 128³ (try 192 or 256 if VRAM allows) |
| `--batch_size` | 1 | Increase if GPU memory allows |
| `--num_workers` | 4 | Parallel data loading processes |

## Hardware Requirements

- **Dev mode (10 skulls):** 16 GB VRAM (e.g., V100, A100)
- **Full dataset, bf=16:** 16-24 GB VRAM
- **Full dataset, bf=32:** 40+ GB VRAM (A100 recommended)
- **Storage:** ~200 GB for full dataset, ~50 GB for dev subset
