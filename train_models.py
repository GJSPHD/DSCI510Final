# train_models.py

pip install tensorflow pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models import build_transformer_model
from dataset import AuroraDataset
import matplotlib.pyplot as plt

def train_model(wide_path="/projects/dsci410_510/Aurora/widegpsptsd.csv", 
                long_path="/projects/dsci410_510/Aurora/longgpsliwc.csv", 
                epochs=150, batch_size=32):
    dataset = AuroraDataset(wide_path, long_path)
    X_ts_train, X_static_train, y_train = dataset.get_train_data()
    X_ts_val, X_static_val, y_val = dataset.get_val_data()
    X_ts_test, X_static_test, y_test = dataset.get_test_data()

    model = build_transformer_model(X_ts_train.shape, X_static_train.shape, y_train.shape[1])

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6)

    history = model.fit(
        [X_ts_train, X_static_train], y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=([X_ts_val, X_static_val], y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    train_loss, train_mae = model.evaluate([X_ts_train, X_static_train], y_train, verbose=0)
    val_loss, val_mae = model.evaluate([X_ts_val, X_static_val], y_val, verbose=0)
    test_loss, test_mae = model.evaluate([X_ts_test, X_static_test], y_test, verbose=0)
    print(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()

    y_pred_test = model.predict([X_ts_test, X_static_test], verbose=0)
    test_results = pd.DataFrame({
        "PID": dataset.test_wide["PID"].values,
        "WK8_PCL5_RS_true": y_test[:, 0], "WK8_PCL5_RS_pred": y_pred_test[:, 0],
        "M3_PCL5_RS_true": y_test[:, 1], "M3_PCL5_RS_pred": y_pred_test[:, 1],
        "M6_PCL5_RS_true": y_test[:, 2], "M6_PCL5_RS_pred": y_pred_test[:, 2]
    })
    test_results.to_csv("test_predictions.csv", index=False)

    model.save("transformer_model.h5")

if __name__ == "__main__":
    train_model()