import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
import h5py
import os

# Load the VGG19 model and trained weights
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Path to the model weights
file_path = os.path.join(
    '/media/dhruv/Local Disk/X__ray/Model-1-20250121T123703Z-001/Model-1/vgg_unfrozen.h5'
)

# Verify the file exists and load weights
if os.path.exists(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print("File loaded successfully!")
        model_03.load_weights(file_path)
        print("Model weights loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
else:
    st.error(f"File not found: {file_path}")


# Helper functions
def get_className(classNo):
    """Return the class name based on the class number."""
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"


def getResult(img):
    """Process the image and return the prediction result."""
    try:
        # Open the image
        image = Image.open(img)
        image = image.resize((128, 128))  # Resize to match input shape
        image = np.array(image)

        # Convert grayscale to RGB if necessary
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize the image and add batch dimension
        image = image / 255.0
        input_img = np.expand_dims(image, axis=0)

        # Predict the result
        result = model_03.predict(input_img)
        result01 = np.argmax(result, axis=1)
        return result01
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# Streamlit App Interface
st.title("Pneumonia Detection from X-ray Images")
st.write("Upload an X-ray image to check if it shows signs of pneumonia or is normal.")

# Upload file widget
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.write("")
    st.write("Processing the image...")

    # Get the prediction result
    try:
        prediction = getResult(uploaded_file)
        if prediction is not None:
            class_name = get_className(prediction[0])
            st.write(f"### Prediction: {class_name}")
        else:
            st.error("Failed to process the image for prediction.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
