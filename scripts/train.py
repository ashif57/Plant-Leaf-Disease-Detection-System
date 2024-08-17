# scripts/train.py


import os
from data_loader import load_data
from model import build_model

def main():
    data_dir = 'data/rice_leaf_diseases'  # Updated path
    img_size = (224, 224)
    batch_size = 32
    epochs = 20  # You can adjust the number of epochs
    input_shape = img_size + (3,)
    
    train_generator, validation_generator = load_data(data_dir, img_size, batch_size)
    
    model = build_model(input_shape, num_classes=len(train_generator.class_indices))
    
    model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    
    model.save('models/plant_disease_model.h5')

""" you can change the path you want to save your model and i didn't push my model in github since the space is so high """


if __name__ == '__main__':
    main()
