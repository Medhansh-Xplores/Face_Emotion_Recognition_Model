from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, train_gen, val_gen, epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping, lr_scheduler]
    )
    return history