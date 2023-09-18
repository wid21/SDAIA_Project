import streamlit as st
import pandas as pd
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("url_goes_here")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)
#Add title
st.title("SALECK")
# Load the model
model = load_model('new_model_denseNet169.h5')

# Define the class labels
class_labels = ['Anomaly', 'Normal', 'Traffic']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to match your model's input size
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    return image

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    # Display the result
    result = f"Prediction: {predicted_class_label}"
    st.write(result)