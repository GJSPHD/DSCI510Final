import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Concatenate, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
import warnings

 y_pred_test = model.predict([X_ts_test, X_static_test], verbose=0)
    test_results = pd.DataFrame({
        "PID": test_wide["PID"],
        **{f"{col}_true": y_test[:, i] for i, col in enumerate(targets)},
        **{f"{col}_pred": y_pred_test[:, i] for i, col in enumerate(targets)}
    })
    test_results.to_csv("/projects/dsci410_510/Aurora/test_predictions.csv", index=False)

# Model Evaluation
    print("\n--- Model Evaluation ---")
    for i, target in enumerate(targets):
        mae = np.mean(np.abs(y_test[:, i] - y_pred_test[:, i]))
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test[:, i]))
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        print(f"{target} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    print("\n--- Classification Metrics (PCL-5 > 33) ---")
    y_test_bin = (y_test > 33).astype(int)
    y_pred_bin = (y_pred_test > 33).astype(int)
    print(classification_report(y_test_bin, y_pred_bin, target_names=targets))

    # Print ALL PID, predicted, and actual PCL-5 scores from test_results
    print("\n--- All Test Predictions ---")
    print(f"Total test set size: {len(test_results)} rows")
    for i in range(len(test_results)):
        pid = test_results["PID"].iloc[i]
        print(f"\nPID: {pid}")
        for target in targets:
            actual = test_results[f"{target}_true"].iloc[i]
            pred = test_results[f"{target}_pred"].iloc[i]
            print(f"{target} - Actual: {actual:.2f}, Predicted: {pred:.2f}")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the CSV file directly
csv_path = "/projects/dsci410_510/Aurora/test_predictions.csv"
df = pd.read_csv(csv_path)

# Define targets and rename columns for consistency
targets = ["WK8_PCL5_RS", "M3_PCL5_RS", "M6_PCL5_RS"]
actual_cols = [f"{target}_true" for target in targets]
pred_cols = [f"{target}_pred" for target in targets]

# Summary Statistics
summary_stats = {}
for target in targets:
    actual = df[f"{target}_true"]
    pred = df[f"{target}_pred"]
    summary_stats[target] = {
        "Mean Actual": actual.mean(),
        "Mean Predicted": pred.mean(),
        "Std Actual": actual.std(),
        "Std Predicted": pred.std(),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R²": r2_score(actual, pred)
    }

# Print Summary Statistics
print("--- Summary Statistics ---")
for target, stats in summary_stats.items():
    print(f"\n{target}:")
    for metric, value in stats.items():
        print(f"  {metric}: {value:.2f}")

# Generate Scatter Plot
plt.figure(figsize=(15, 5))
for i, target in enumerate(targets, 1):
    plt.subplot(1, 3, i)
    plt.scatter(df[f"{target}_true"], df[f"{target}_pred"], alpha=0.5, label="Predictions")
    plt.plot([0, 80], [0, 80], 'r--', label="Perfect Prediction")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target} (MAE: {summary_stats[target]['MAE']:.2f})")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
