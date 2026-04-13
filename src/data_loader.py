import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def get_data_generators():
    """
    Creates and returns train, validation, and test data generators.
    """
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and Test data generator (only rescaling)
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Load data from directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = valid_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
