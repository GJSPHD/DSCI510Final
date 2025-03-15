import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU warnings
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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_long, train_wide, val_long, val_wide, test_long, test_wide = load_and_split_data()
    X_ts_train, X_static_train, y_train, mask_train, ts_features, static_features, targets = preprocess_data(train_long, train_wide)
    X_ts_val, X_static_val, y_val, mask_val, _, _, _ = preprocess_data(val_long, val_wide)
    X_ts_test, X_static_test, y_test, mask_test, _, _, _ = preprocess_data(test_long, test_wide)

    print(f"y_train mean: {y_train.mean(axis=0)}, SD: {y_train.std(axis=0)}")
    print(f"y_val mean: {y_val.mean(axis=0)}, SD: {y_val.std(axis=0)}")
    print(f"y_test mean: {y_test.mean(axis=0)}, SD: {y_test.std(axis=0)}")

    model = build_transformer_model(X_ts_train.shape, X_static_train.shape, y_train.shape[1])
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    history = model.fit(
        [X_ts_train, X_static_train], y_train,
        epochs=21, batch_size=32, validation_data=([X_ts_val, X_static_val], y_val),
        callbacks=[early_stopping, reduce_lr], verbose=1
    )

    train_loss, train_mae = model.evaluate([X_ts_train, X_static_train], y_train, verbose=0)
    val_loss, val_mae = model.evaluate([X_ts_val, X_static_val], y_val, verbose=0)
    test_loss, test_mae = model.evaluate([X_ts_test, X_static_test], y_test, verbose=0)
    print(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    y_pred_test = model.predict([X_ts_test, X_static_test], verbose=0)
    test_results = pd.DataFrame({
        "PID": test_wide["PID"],
        **{f"{col}_true": y_test[:, i] for i, col in enumerate(targets)},
        **{f"{col}_pred": y_pred_test[:, i] for i, col in enumerate(targets)}
    })
    test_results.to_csv("/projects/dsci410_510/Aurora/test_predictions.csv", index=False)
