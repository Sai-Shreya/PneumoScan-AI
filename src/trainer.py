import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.config import EPOCHS, LEARNING_RATE
import numpy as np

def train_model(model, train_gen, val_gen, model_path):
    """
    Trains the given model with specified data generators.
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    # Calculate class weights for imbalance handling
    labels = train_gen.classes
    class_counts = np.bincount(labels)
    total = len(labels)
    weight_for_0 = (1 / class_counts[0]) * (total / 2.0)
    weight_for_1 = (1 / class_counts[1]) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f"Class Weights: {class_weight}")

    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight
    )

    return history
