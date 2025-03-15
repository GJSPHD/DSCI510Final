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


# --- MODEL PY ---
def build_transformer_model(X_ts_shape, X_static_shape, num_targets):
    ts_input = Input(shape=(X_ts_shape[1], X_ts_shape[2]), name="ts_input")
    static_input = Input(shape=(X_static_shape[1],), name="static_input")

    x = MultiHeadAttention(num_heads=8, key_dim=64)(ts_input, ts_input)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(ts_input + x)
    ff = Dense(X_ts_shape[2], activation="relu")(x)
    ff = Dropout(0.1)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)

    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x + x)
    ff = Dense(X_ts_shape[2], activation="relu")(x)
    ff = Dropout(0.1)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)

    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x + x)
    ff = Dense(X_ts_shape[2], activation="relu")(x)
    ff = Dropout(0.1)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)

    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x + x)
    ff = Dense(X_ts_shape[2], activation="relu")(x)
    ff = Dropout(0.1)(ff)
    ts_output = LayerNormalization(epsilon=1e-6)(x + ff)
    ts_flat = Dense(32, activation="relu")(ts_output[:, -1, :])

    static_dense = Dense(64, activation="relu")(static_input)
    static_weights = Dense(64, activation="sigmoid")(static_dense)
    static_weighted = Multiply()([static_dense, static_weights])
    static_dropout = Dropout(0.1)(static_weighted)

    combined = Concatenate()([ts_flat, static_dropout])
    combined_dense = Dense(64, activation="relu")(combined)
    combined_dropout = Dropout(0.1)(combined_dense)
    final_dense = Dense(16, activation="relu")(combined_dropout)
    output = Dense(num_targets, name="output")(final_dense)

    model = Model(inputs=[ts_input, static_input], outputs=output)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=5e-5), loss="mse", metrics=["mae"])
    return model
