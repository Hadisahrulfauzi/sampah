import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import numpy as np
import os

# Define constants
IMG_SIZE = 224  # Image size for resizing
CLASSES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']  # Replace with your actual class names
MODEL_PATH = "modelResNet50_model.pth"  # Path to the saved model

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize the image
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Load the model
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # Adjust the final layer to match the number of classes

    # Load the saved model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to classify an uploaded image
def classify_image(image, model):
    try:
        # Convert the image to RGB and apply transformations
        img = Image.open(image).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get the predicted class
        predicted_class = predicted.item()
        predicted_class_name = CLASSES[predicted_class]

        return img, predicted_class_name

    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None

# Streamlit app
def main():
    st.title("Image Classification with ResNet50")

    # Upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load the model
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Classify the image
        img, predicted_class_name = classify_image(uploaded_file, model)

        if img is not None:
            # Display the result
            st.write(f"Predicted Class: {predicted_class_name}")
            
            # Show the image with Matplotlib (optional)
            fig, ax = plt.subplots()
            ax.imshow(np.array(img))
            ax.set_title(f"Predicted Class: {predicted_class_name}")
            ax.axis("off")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
