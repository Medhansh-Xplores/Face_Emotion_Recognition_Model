import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(48, 48), batch_size=64):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', color_mode='grayscale'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', color_mode='grayscale'
    )

    return train_generator, val_generator
