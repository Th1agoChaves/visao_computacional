from tensorflow import keras

def train_model(model, X_train, y_train, epochs=50, batch_size=64, validation_split=0.2):
    """Treina o modelo com EarlyStopping e retorna o histórico."""
    early_stop= keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
    )
    return history