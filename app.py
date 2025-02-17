import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("./model/skin_disease_classifier_V1.h5")

# Define class labels
class_labels = [
    "Melanoma",
    "Vascular lesion",
    "Melanocytic nevus",
    "Actinic keratosis",
    "Squamous cell carcinoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Dermatofibroma"
]

def preprocess_image(img):
    """
    Preprocess the uploaded image to match the input format required by the model.
    - Resizes the image to (64, 64)
    - Converts it to an array
    - Normalizes pixel values between 0 and 1
    - Expands dimensions to fit the model's expected input shape
    """
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_image(img):
    """
    Runs the preprocessed image through the model to obtain predictions.
    - Identifies the class with the highest probability
    - Returns the predicted class name and confidence score
    """
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence

st.title("Skin Disease Classifier")
st.write("""
### About This App
This web application uses a deep learning model to classify skin conditions. 
Simply upload an image of a skin lesion, and the model will analyze it to provide a predicted diagnosis.

### Supported Skin Conditions:
1. **Melanoma** - A serious form of skin cancer that can spread to other parts of the body.
2. **Vascular lesion** - An abnormality of blood vessels, often appearing as red or purple skin discoloration.
3. **Melanocytic nevus** - A mole or birthmark that contains pigment-producing cells.
4. **Actinic keratosis** - A rough, scaly patch on the skin caused by sun exposure.
5. **Squamous cell carcinoma** - A common form of skin cancer arising from squamous cells.
6. **Benign keratosis** - A non-cancerous skin growth that often appears as a wart-like lesion.
7. **Basal cell carcinoma** - A type of skin cancer that begins in the basal cells.
8. **Dermatofibroma** - A benign skin nodule commonly found on the legs.
""")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image",  use_container_width=True)
    st.write("### Processing image... Please wait.")
    
    # Get prediction
    label, confidence = predict_image(img)
    
    # Display results~
    st.success(f"### Prediction: {label}")
    st.info(f"### Confidence: {confidence:.2f}")
    
    st.write("""
### Important Note:
- This prediction is **not a substitute for professional medical advice**.
- If you suspect a skin condition, please consult a dermatologist for a thorough evaluation.
""")