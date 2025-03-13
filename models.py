# models.py

pip install tensorflow pandas numpy scikit-learn matplotlib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Concatenate, Multiply
from tensorflow.keras.optimizers import AdamW

def build_transformer_model(X_ts_shape, X_static_shape, num_targets):
    """
    Build a transformer model for PTSD prediction.
    
    Args:
        X_ts_shape (tuple): Shape of time-series input (n_samples, 181, 7)
        X_static_shape (tuple): Shape of static input (n_samples, n_static_features)
        num_targets (int): Number of output targets (3 for WK8, M3, M6 PCL-5 scores)
    
    Returns:
        Model: Compiled Keras model
    """
    ts_input = Input(shape=(X_ts_shape[1], X_ts_shape[2]), name="ts_input")  # e.g., (181, 7)
    static_input = Input(shape=(X_static_shape[1],), name="static_input")     # e.g., (50+)

    # Transformer encoder (4 blocks)
    x = ts_input
    for _ in range(4):
        attn = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        attn = Dropout(0.1)(attn)
        x = LayerNormalization(epsilon=1e-6)(x + attn)
        ff = Dense(X_ts_shape[2], activation="relu")(x)  # Matches input feature dim (7)
        ff = Dropout(0.1)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)

    # Use last timestep
    ts_output = x[:, -1, :]
    ts_flat = Dense(32, activation="relu")(ts_output)

    # Static feature processing
    static_dense = Dense(64, activation="relu")(static_input)
    static_weights = Dense(64, activation="sigmoid")(static_dense)
    static_weighted = Multiply()([static_dense, static_weights])
    static_dropout = Dropout(0.1)(static_weighted)

    # Combine features
    combined = Concatenate()([ts_flat, static_dropout])
    combined_dense = Dense(64, activation="relu")(combined)
    combined_dropout = Dropout(0.1)(combined_dense)
    final_dense = Dense(16, activation="relu")(combined_dropout)
    output = Dense(num_targets, name="output")(final_dense)

    model = Model(inputs=[ts_input, static_input], outputs=output)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=5e-5), 
                  loss="mse", 
                  metrics=["mae"])
    return model

if __name__ == "__main__":
    # Test with expected shapes
    model = build_transformer_model((None, 181, 7), (None, 50), 3)
    model.summary()