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