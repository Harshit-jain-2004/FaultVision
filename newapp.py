import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLOv8 model directly from the local path
MODEL_PATH = r"C:\Users\HP\Desktop\PCB-defect-detection\runs\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)

# Streamlit app title and description
st.title("PCB Defect Detection")
st.write("Upload an image and the model will detect PCB defects.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to a format suitable for the model (e.g., numpy array)
    img = np.array(image)

    # Perform inference
    results = model(img)

    # Draw the results on the image
    result_img = results[0].plot()  # results[0].plot() returns an image with predictions

    # Convert the result image to RGB format for display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Display the "Result" heading
    st.subheader("Result")

    # Display the predicted image
    st.image(result_img_rgb, caption='Predicted Image with Defects', use_column_width=True)

    # Save button logic
    if st.button("Save"):
        save_path = r"C:\Users\HP\Desktop\PCB-defect-detection\results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, "predicted_image.png")
        cv2.imwrite(file_name, cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR))  # Save in BGR format
        st.success(f"Image saved at {file_name}")
