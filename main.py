# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:26:22 2023

@author: yasha
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Load the saved model
loaded_model = load_model("cats_dogs_model.h5")

# Streamlit app
st.title("Cat vs Dog Classifier")
st.write("Upload an image to classify whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(256,256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = loaded_model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    # Display the uploaded image and the prediction result
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {result}")
