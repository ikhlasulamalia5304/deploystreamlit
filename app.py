import streamlit as st
import pandas as pd
import joblib 
import numpy as np

# --- 1. MUAT MODEL (Pastikan nama file ini sudah ada di folder Anda) ---
MODEL_FILE = 'model_xgb.pkl' # <-- Nama file yang BENAR

try:
    model = joblib.load(MODEL_FILE)
    # Jika Anda menggunakan scaler atau encoder (perlu dicek di notebook)
    # scaler = joblib.load('scaler.pkl') 
    st.sidebar.success("Model dan aset berhasil dimuat!") 
except FileNotFoundError:
    st.error(f"Error: File model {MODEL_FILE} tidak ditemukan di root folder. Deploy akan gagal.")
    st.stop()

# --- 2. TITTLE DAN DESKRIPSI APLIKASI ---
st.title("🩸 Prediksi Klasifikasi Anemia dengan XGBoost")
st.write("Masukkan nilai-nilai pasien untuk memprediksi risiko Anemia.")

# --- 3. INPUT DARI PENGGUNA ---

# Sesuaikan dengan semua fitur yang digunakan model XGBoost Anda, 
# dan pastikan URUTANNYA SAMA PERSIS dengan saat pelatihan!

st.header("Data Pasien")

# Contoh Input (Ganti dengan fitur aktual dari model Anda)
hemoglobin = st.number_input('1. Hemoglobin (g/dL)', min_value=0.0, max_value=20.0, value=12.0)
mch = st.number_input('2. MCH (pg)', min_value=0.0, max_value=100.0, value=27.0)
mcv = st.number_input('3. MCV (fL)', min_value=0.0, max_value=150.0, value=85.0)
gender = st.selectbox('4. Jenis Kelamin', ['Wanita', 'Pria'])

# Konversi Gender ke 0 atau 1 (sesuaikan dengan cara Anda melakukan One-Hot Encoding/Label Encoding)
gender_encoded = 1 if gender == 'Pria' else 0


# --- 4. PREDIKSI (Saat tombol ditekan) ---
if st.button('Prediksi Risiko Anemia'):
    
    # Kumpulkan input pengguna sesuai URUTAN FITUR PELATIHAN
    data_input_list = [gender_encoded, hemoglobin, mch, mcv] # CONTOH URUTAN
    data_input = np.array([data_input_list])
    
    # 4a. Jika Anda menggunakan scaler, terapkan di sini
    # data_input_scaled = scaler.transform(data_input)
    
    # 4b. Lakukan Prediksi
    prediksi = model.predict(data_input) 
    
    # 4c. Tampilkan Hasil
    st.subheader("Hasil Prediksi:")
    if prediksi[0] == 1:
        st.error("🚨 ANEMIA: Risiko tinggi terdeteksi.")
    else:
        st.success("✅ TIDAK ANEMIA: Risiko rendah.")