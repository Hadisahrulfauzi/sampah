import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Path model
model_path = 'D:/PCD/modelResNet101_model.pth'

# Memuat model yang sudah dilatih
def load_model(model_path):
    # Membuat model ResNet101
    model = models.resnet101(pretrained=False)  # Gunakan pretrained=False karena model sudah dilatih
    model.fc = torch.nn.Linear(model.fc.in_features, 9)  # Sesuaikan output dengan jumlah kelas

    if os.path.exists(model_path):
        # Memuat model ke perangkat yang sesuai (CPU atau GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)  # Pastikan model berada di device yang benar (GPU atau CPU)
        model.eval()  # Set the model to evaluation mode
        return model, device
    else:
        st.error(f"Model tidak ditemukan di {model_path}")
        return None, None

# Memuat model
model, device = load_model(model_path)

# Jika model gagal dimuat, hentikan eksekusi
if model is None:
    st.stop()

# Kelas yang diprediksi oleh model
classes = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 
           'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Transformasi untuk preprocessing gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize gambar agar sesuai dengan ukuran input model
    transforms.ToTensor(),  # Konversi gambar ke tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi untuk ResNet
])

# Fungsi untuk memproses gambar input
def preprocess_image(img):
    img = img.convert("RGB")  # Convert to RGB jika tidak
    img_tensor = transform(img)  # Terapkan transformasi
    img_tensor = img_tensor.unsqueeze(0)  # Menambahkan dimensi batch
    return img_tensor

# Fungsi untuk memprediksi gambar
def predict_image(img_tensor, device):
    img_tensor = img_tensor.to(device)  # Pastikan tensor berada di perangkat yang benar (CPU atau GPU)
    
    with torch.no_grad():
        outputs = model(img_tensor)  # Prediksi menggunakan model
        _, class_idx = torch.max(outputs, 1)  # Dapatkan indeks kelas yang diprediksi
        predicted_class = classes[class_idx.item()]  # Dapatkan nama kelas
        probability = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()  # Probabilitas kelas terprediksi
        return predicted_class, probability

# Menyimpan riwayat ke session state jika belum ada
if "history" not in st.session_state:
    st.session_state.history = []

# Halaman "Kamera" untuk mengunggah gambar
if menu == "Kamera":
    st.header("Unggah Gambar untuk Klasifikasi")
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            # Preprocess gambar dan buat prediksi
            img_tensor = preprocess_image(image)
            predicted_class, probability = predict_image(img_tensor, device)
            st.write(f"Kelas yang diprediksi: {predicted_class}")
            st.write(f"Probabilitas: {probability * 100:.2f}%")

            # Menyimpan riwayat klasifikasi
            st.session_state.history.append({
                "image": uploaded_file.name,
                "predicted_class": predicted_class,
                "probability": probability
            })
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

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
