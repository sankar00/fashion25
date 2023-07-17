import io
import tensorflow as tf
from keras.utils.image_utils import img_to_array, load_img
import numpy as np
import pickle

categoricals = ["Bags", "Belts", "Bottomwear", "Dress", "Eyewear", "Flip Flops", "Fragrance", "Innerwear",
                "Jewellery", "Lips", "Loungewear and Nightwear", "Makeup", "Nails", "Sandal", "Saree", "Shoes",
                "Socks", "Topwear", "Wallets", "Watches"]

'''def load_model():
    with open('model (1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model'''
from tensorflow.keras.models import load_model

model = load_model(filepath='fastapi.h5')
def load_model(filepath):
    model = load_model(filepath)
    return model
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_category(image_content):
    image = load_img(io.BytesIO(image_content))
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_category_index = np.argmax(predictions)
    predicted_category = categoricals[predicted_category_index]
    return predicted_category

