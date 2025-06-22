import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi HIV", layout="wide")
st.title("Prediksi Risiko HIV (Pretrained Random Forest Model)")
st.markdown("**Kelompok 6 - Universitas Nusa Putra**")

# Load model dan encoders
@st.cache_resource
def load_model_and_encoders():
    with open("randomforest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_model_and_encoders()

# Upload dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Unggah file CSV dataset HIV", type="csv")

# Preprocessing label encoding
def preprocess_input(df, label_encoders):
    df = df.copy()
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    return df

# Proses jika file diunggah
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")
    st.write("Contoh data:", df.head())

    if 'Result' not in df.columns:
        st.error("❌ Kolom 'Result' tidak ditemukan. Pastikan dataset memiliki kolom ini.")
        st.stop()

    df = df[df['Result'].isin([0, 1])].reset_index(drop=True)
    X = df.drop('Result', axis=1)
    y_true = df['Result']

    try:
        X_encoded = preprocess_input(X, label_encoders)

        # Pastikan urutan kolom sesuai model
        expected_cols = model.feature_names_in_
        missing_cols = [col for col in expected_cols if col not in X_encoded.columns]
        extra_cols = [col for col in X_encoded.columns if col not in expected_cols]

        if missing_cols:
            st.error(f"❌ Dataset Anda kekurangan kolom berikut yang dibutuhkan model: {missing_cols}")
            st.stop()

        if extra_cols:
            st.warning(f"⚠️ Dataset Anda memiliki kolom tambahan yang tidak digunakan oleh model: {extra_cols}")

        X_encoded = X_encoded[expected_cols]

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat preprocessing: {e}")
        st.stop()

    # Prediksi
    y_pred = model.predict(X_encoded)

    # Evaluasi
    st.subheader("📊 Evaluasi Model")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    eval_df = pd.DataFrame({
        'Metrik': ['Akurasi', 'Precision', 'Recall', 'F1-score'],
        'Hasil (%)': [f"{acc * 100:.2f}%", f"{prec * 100:.2f}%", f"{rec * 100:.2f}%", f"{f1 * 100:.2f}%"]
    })
    st.dataframe(eval_df, use_container_width=True)

    csv_eval = eval_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Evaluasi", csv_eval, file_name="evaluasi_model.csv", mime="text/csv")

    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    # Hasil prediksi
    st.subheader("📥 Unduh Hasil Prediksi")
    df_out = df.copy()
    df_out['Prediksi'] = y_pred
    st.write("Contoh hasil prediksi:", df_out.head())

    csv_out = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Prediksi", csv_out, file_name="hasil_prediksi.csv", mime="text/csv")
