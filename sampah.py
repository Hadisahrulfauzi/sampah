import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'D:/PCD/modelResNet101_model.pth'
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di {model_path}")
else:
    # Load the PyTorch model (ResNet101 in this case)
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Memuat nama kelas dari model (disesuaikan dengan jumlah kelas model)
    classes = ['Cardboard','Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']
    
    # Preprocessing the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
    ])

    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        img = img.convert("RGB")  # Convert to RGB if not
        img_tensor = transform(img)  # Apply the transformations
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor

    # Fungsi untuk memprediksi gambar
    def predict_image(img_tensor):
        with torch.no_grad():
            outputs = model(img_tensor)  # Predict using the model
            _, class_idx = torch.max(outputs, 1)  # Get the predicted class index
            predicted_class = classes[class_idx.item()]  # Get the class name
            return predicted_class, torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()  # Return class and probability

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Halaman "Kamera" untuk mengunggah gambar
    if menu == "Kamera":
        st.header("Unggah Gambar untuk Klasifikasi")
        uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            # Preprocess image and make prediction
            img_tensor = preprocess_image(image)
            predicted_class, probability = predict_image(img_tensor)
            st.write(f"Kelas yang diprediksi: {predicted_class}")
            st.write(f"Probabilitas: {probability * 100:.2f}%")

            # Menyimpan riwayat klasifikasi
            st.session_state.history.append({
                "image": uploaded_file.name,
                "predicted_class": predicted_class,
                "probability": probability
            })

    # Halaman "Riwayat" untuk melihat riwayat klasifikasi
    elif menu == "Riwayat":
        st.header("Riwayat Klasifikasi")
        if st.session_state.history:
            for entry in st.session_state.history:
                st.write(f"*Gambar*: {entry['image']}")
                st.write(f"*Kelas yang diprediksi*: {entry['predicted_class']}")
                st.write(f"*Probabilitas*: {entry['probability'] * 100:.2f}%")
        else:
            st.write("Tidak ada riwayat klasifikasi.")
