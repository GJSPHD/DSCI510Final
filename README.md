# DSCI 510 Final Project - Aurora

Transformer-based neural network for predicting outcomes using time-series and static data. Workflow in Python scripts (`step1` to `step4`); `AuroraDSCI510FinalFull.ipynb` as fallback.

## Contents
- `best_model_weights.h5`: Pre-trained weights (best validation).
- `AuroraDSCI510FinalFull.ipynb`: Full workflow backup.
- `step1_datasets.py`: Load, split, preprocess datasets.
- `step2_models.py`: Define transformer model.
- `step3_testdata.py`: Train model, save predictions/weights.
- `step4_modeleval.py`: Evaluate model, generate metrics/plots.
- `README.md`: This file.
- `.git`: Git metadata.

## How to Use

### Prerequisites
- Python 3.x: `tensorflow`, `pandas`, `numpy`, `sklearn`, `matplotlib`.
- `pip install tensorflow pandas numpy scikit-learn matplotlib`.
- Datasets: `widegpsptsd.csv`, `longgpsliwc.csv` in `/projects/dsci410_510/Aurora/`.
- Write access: `/projects/dsci410_510/Aurora/`.

### Python Scripts
1. `python step1_datasets.py`: Load, split, preprocess data.
2. `python step2_models.py`: Define model.
3. `python step3_testdata.py`: Train, save `test_predictions.csv`, weights.
4. `python step4_modeleval.py`: Evaluate, plot results.

### Backup Notebook
- Open `AuroraDSCI510FinalFull.ipynb`.
- Run the cell
- Outputs in `/projects/dsci410_510/Aurora/`.

## Steps
- `step1_datasets.py`: Load, split, preprocess (impute, scale, engineer features).
- `step2_models.py`: Define transformer (multi-head attention, MSE loss).
- `step3_testdata.py`: Train with early stopping, save outputs.
- `step4_modeleval.py`: Compute MAE, RMSE, R², PCL-5 > 33, visualize.

## Notes
- Run from `/projects/dsci410_510/Aurora/` accessible directory.
- Notebook is fallback for script issues.
- Adjust paths if needed.
