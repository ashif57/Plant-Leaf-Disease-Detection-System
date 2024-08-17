# scripts/data_loader.py
""" dataset i used: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
