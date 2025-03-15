
# DSCI 510 Final Project - Aurora

## Documentation

### Introduction
This project creates a transformer-based neural network to predict PCL-5 scores (indicating PTSD severity) at 8 weeks, 3 months, and 6 months post-trauma using time-series and static data from the Aurora dataset. The purpose is to model longitudinal trauma outcomes by integrating demographic, trauma-related, GPS, and text sentiment data, aiding in early PTSD identification and intervention.

### Dataset
- **Source**: Advancing Understanding of Recovery After Trauma (Aurora) dataset
- **Wide Data**: `/projects/dsci410_510/Aurora/widegpsptsd.csv`
  - Demographics (e.g., age, gender, race) and trauma-related measures (e.g., PCL-5, PSS10) collected at multiple time points
- **Long Data**: `/projects/dsci410_510/Aurora/longgpsliwc.csv`
  - Daily observations from GPS data (e.g., distance traveled) and LIWC/semantic sentiment analysis of text messages (e.g., positive emotion, anger)
- **Creation**: Wide data from Aurora; long data processed from raw GPS and text into daily aggregates using LIWC tools and spatial analysis. Cleaning/merging in `PTSD_510_Loading_Cleaning.ipynb`
- **Original Data Source**: https://www.med.unc.edu/itr/aurora-study/

### Instructions on How to Train the Model
- **Prerequisites**: Python 3.x with `tensorflow`, `pandas`, `numpy`, `sklearn`, `matplotlib`. Install: `pip install tensorflow pandas numpy scikit-learn matplotlib`. Datasets in `/projects/dsci410_510/Aurora/`. Write access to `/projects/dsci410_510/Aurora/`.
- **Steps**:
  1. `python step1_datasets.py`: Load, split, preprocess wide/long data
  2. `python step2_models.py`: Define transformer model
  3. `python step3_modeltraining.py`: Train with early stopping, save `test_predictions.csv`, `best_model_weights.h5`, `final_model_weights.h5`. Uses AdamW optimizer, MSE loss, 21 epochs, batch size 32
  4. `python step4_test_data_and_modeleval.py`: Evaluate results
- **Backup Option**: Open `AuroraDSCI510FinalFull.ipynb`, run all cells
- **Load Trained Weights**:
  from step2_models import build_transformer_model
  model = build_transformer_model(X_ts_shape, X_static_shape, 3)
  model.load_weights("/projects/dsci410_510/Aurora/Github_clone/best_model_weights.h5")
 
### Contents
- `best_model_weights.h5`: Pre-trained weights (best validation) - 
- `AuroraDSCI510FinalFull.ipynb`: Full workflow backup - 
- `step1_datasets.py`: Load, split, preprocess datasets -
- `step2_models.py`: Define transformer model -
- `step3_modeltraining.py`: Train model, save predictions/weights
- `step4_test_data_and_modeleval.py`: Evaluate model, generate metrics/plots 
- `PTSD_510_Loading_Cleaning.ipynb`: Data cleaning 
- `MAE rates.png`: Graph of MAE (scatter plots of actual vs. predicted PCL-5 scores)
- `training and loss validation.png`: Graph of training and validation loss 
- `README.md`: This file - 

### GitHub Repository
- https://github.com/GJSPHD/DSCI510Final

## Results

### Metrics
- **WK8_PCL5_RS**: MAE: 10.99, RMSE: 14.31, R²: 0.36
- **M3_PCL5_RS**: MAE: 12.07, RMSE: 15.55, R²: 0.25
- **M6_PCL5_RS**: MAE: 11.77, RMSE: 15.19, R²: 0.29
- **Classification Metrics (PCL-5 > 33)**:
```
            precision  recall  f1-score  support
WK8_PCL5_RS    0.81     0.69    0.75      32
M3_PCL5_RS     0.78     0.58    0.67      31
M6_PCL5_RS     0.73     0.55    0.63      29
micro avg      0.78     0.61    0.68      92
macro avg      0.77     0.61    0.68      92
weighted avg   0.78     0.61    0.68      92
samples avg    0.24     0.24    0.23      92
 ```
- **Summary Statistics**:
  - WK8_PCL5_RS: Mean Actual: 32.25, Mean Predicted: 28.37, Std Actual: 17.99, Std Predicted: 13.62
  - M3_PCL5_RS: Mean Actual: 29.90, Mean Predicted: 24.34, Std Actual: 18.09, Std Predicted: 12.18
  - M6_PCL5_RS: Mean Actual: 26.71, Mean Predicted: 23.18, Std Actual: 18.18, Std Predicted: 11.68

### Visualization of Model Predictions
You have to click on the files in the repository to view 
- `MAE rates.png`: Graph of actual vs. predicted PCL-5 scores from `step4_test_data_and_modeleval.py`
- `training and loss validation.png`: Graph of training and validation loss (MSE) over epochs from `step3_modeltraining.py`

### Limitations and Use
- **Limitations**: Moderate R² (0.25-0.36) and MAE (10.99-12.07) show limited accuracy, especially for M3/M6. Small test set (86 rows) and imputation may reduce generalizability. Model underestimates high PCL-5 scores.
- **Use**: Useful for early PTSD screening with Aurora-like data, but needs larger datasets and refinement for precise predictions.

## Notes
- Talapas-- Run from `/projects/dsci410_510/Aurora/`
- Jupyter Notebook  `AuroraDSCI510FinalFull.ipynb` includes full model, evaluation, demo data, and graphs
- Test set size: 86 rows; full predictions in `step4_test_data_and_modeleval.py` output

