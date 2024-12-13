import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'D:\PCD\modelResNet101_model.pth'
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di {model_path}")
else:
    model = tf.keras.models.load_model(model_path, compile=False)

    # Memuat nama kelas dari model (disesuaikan dengan jumlah kelas model)
    classes = ['Cardboard','Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']
    

    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        img = img.resize((224, 224))  # Mengubah ukuran gambar sesuai input model
        img_array = np.array(img) / 255.0  # Normalisasi gambar
        img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
        return img_array

    # Fungsi untuk memprediksi gambar
    def predict_image(img_array):
        preds = model.predict(img_array)  # Prediksi menggunakan model
        class_idx = np.argmax(preds, axis=1)  # Menentukan kelas dengan probabilitas tertinggi
        return classes[class_idx[0]], preds[0][class_idx[0]]  # Mengembalikan kelas dan probabilitas

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.image("https://example.com/path_to_header_image.jpg", use_container_width=True)  # Ganti dengan URL gambar header yang sesuai
    st.title("Deteksi Penyakit Tanaman dengan AI")

    if menu == "Beranda":
        st.markdown("""
        Aplikasi ini menggunakan model TensorFlow untuk mendeteksi penyakit pada tanaman kentang.
        Gunakan menu Kamera untuk mengambil gambar daun tanaman dan memprediksi penyakit yang ada.
        """, unsafe_allow_html=True)

    elif menu == "Kamera":
        # Menampilkan pilihan untuk mengambil gambar menggunakan kamera
        camera_input = st.camera_input("Ambil gambar untuk diprediksi")

        if camera_input is not None:
            # Menampilkan gambar yang diambil
            st.image(camera_input, caption="Gambar yang diambil.", use_container_width=True)

            # Memproses gambar
            img = Image.open(camera_input)
            img_array = preprocess_image(img)

            # Prediksi
            label, confidence = predict_image(img_array)
            st.write(f"Prediksi: {label}")
            st.write(f"Probabilitas: {confidence:.2f}")

            # Menyimpan gambar dan hasil prediksi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append({
                "image": img_bytes,
                "label": label,
                "confidence": confidence
            })

    elif menu == "Riwayat":
        # Menampilkan riwayat hasil prediksi
        if len(st.session_state.history) == 0:
            st.write("Tidak ada riwayat prediksi.")
        else:
            st.write("Riwayat Prediksi Penyakit Tanaman:")

            # Loop untuk menampilkan setiap entri dalam riwayat
            for i, entry in enumerate(st.session_state.history):
                # Menampilkan gambar dari riwayat
                st.image(entry["image"], caption=f"Prediksi {i+1}: {entry['label']} (Probabilitas: {entry['confidence']:.2f})", use_container_width=True)
                st.write(f"Prediksi: {entry['label']}")
                st.write(f"Probabilitas: {entry['confidence']:.2f}")

                # Menambahkan tombol hapus
                if st.button(f"Hapus Prediksi {i+1}", key=f"hapus_{i}"):
                    # Menghapus entri dari riwayat
                    st.session_state.history.pop(i)
                    st.rerun()  # Me-refresh halaman setelah penghapusan
                st.markdown("---")

# Menambahkan CSS kustom untuk mempercantik tampilan
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #6495ED;
            color: white;
            padding: 20px 0;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }
        .css-ffhzg2 {
            font-size: 1.25em;
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)