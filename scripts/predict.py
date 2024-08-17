# scripts/predict.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_image(img_path, model_path, img_size=(224, 224)):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    return predictions

if __name__ == '__main__':
    img_path = r'D:\Task-2\Plant-Leaf-Disease-Detection-System\scripts\__pycache__\testfromweb.jpg'  # Update this path
    model_path = r'D:\Task-2\Plant-Leaf-Disease-Detection-System\models\plant_disease_model.h5'

    # Map class indices to class names
    class_names = {
        0: 'Bacterial leaf blight',
        1: 'Brown spot',
        2: 'Leaf smut'
    }

    predictions = predict_image(img_path, model_path)
    predicted_class = np.argmax(predictions)
    print(f'Predicted class: {class_names[predicted_class]}')
    print(f'Prediction probabilities: {predictions}')
