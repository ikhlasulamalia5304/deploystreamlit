import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from xgboost import XGBClassifier # Wajib di-import untuk memuat model JSON

# --- 1. MUAT MODEL (Gunakan format JSON yang stabil) ---
MODEL_FILE = 'model_xgb.json' 
SCALER_FILE = 'scaler.pkl'

try:
    # 1. Muat Scaler (Joblib aman untuk Scikit-learn scaler)
    scaler = joblib.load(SCALER_FILE)
    
    # 2. Muat Model XGBoost (Gunakan format native JSON)
    model = XGBClassifier() # Inisiasi objek kosong
    model.load_model(MODEL_FILE) # Muat data model dari file JSON

    st.sidebar.success("Model dan scaler berhasil dimuat!") 
except FileNotFoundError:
    st.error(f"Error: Pastikan file model ({MODEL_FILE}) dan scaler ({SCALER_FILE}) sudah di-push.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# --- 2. TITTLE DAN DESKRIPSI APLIKASI ---
st.title("🩸 Prediksi Klasifikasi Anemia dengan XGBoost")
st.write("Masukkan nilai-nilai pasien untuk memprediksi risiko Anemia.")

# --- 3. INPUT DARI PENGGUNA ---

st.header("Data Pasien")

# Urutan Input: Hemoglobin, MCH, MCV, Gender (sesuaikan dengan urutan kolom data Anda)
hemoglobin = st.number_input('1. Hemoglobin (g/dL)', min_value=0.0, max_value=20.0, value=12.0)
mch = st.number_input('2. MCH (pg)', min_value=0.0, max_value=100.0, value=27.0)
mcv = st.number_input('3. MCV (fL)', min_value=0.0, max_value=150.0, value=85.0)
gender = st.selectbox('4. Jenis Kelamin', ['Wanita', 'Pria'])

# Konversi Gender ke 0 (Wanita) atau 1 (Pria)
# Pria = 1, Wanita = 0 (Asumsi Label Encoding Anda)
gender_encoded = 1 if gender == 'Pria' else 0


# --- 4. PREDIKSI (Saat tombol ditekan) ---
if st.button('Prediksi Risiko Anemia'):
    
    # 4a. FEATURE ENGINEERING: Hitung Hb/MCV Ratio (Wajib!)
    # Berdasarkan jurnal Anda, fitur ini sangat penting
    try:
        hb_mcv_ratio = hemoglobin / mcv
    except ZeroDivisionError:
        st.error("MCV tidak boleh nol.")
        st.stop()
    
    # Kumpulkan input pengguna sesuai URUTAN FITUR PELATIHAN
    # URUTAN HARUS SAMA DENGAN URUTAN FITUR DI NOTEBOOK ANDA (termasuk Ratio)
    # Asumsi urutan fitur Anda: [Gender, Hemoglobin, MCH, MCV, Hb/MCV Ratio]
    data_input_list = [gender_encoded, hemoglobin, mch, mcv, hb_mcv_ratio] 
    data_input = np.array([data_input_list])
    
    # 4b. SCALING DATA: Terapkan scaler
    # Scaling dilakukan pada SEMUA fitur, termasuk fitur yang baru dibuat.
    data_input_scaled = scaler.transform(data_input)
    
    # 4c. Lakukan Prediksi
    prediksi = model.predict(data_input_scaled) 
    prediksi_prob = model.predict_proba(data_input_scaled)
    
    # 4d. Tampilkan Hasil
    st.subheader("Hasil Prediksi:")
    if prediksi[0] == 1:
        st.error(f"🚨 ANEMIA: Risiko tinggi terdeteksi (Probabilitas: {prediksi_prob[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ TIDAK ANEMIA: Risiko rendah (Probabilitas: {prediksi_prob[0][0]*100:.2f}%)")
    
    st.caption("Prediksi dilakukan oleh model XGBoost yang dilatih dengan fitur rasio Hb/MCV.")