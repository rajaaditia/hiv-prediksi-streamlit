import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Prediksi HIV", layout="wide")
st.title("Prediksi Risiko HIV (Pretrained Random Forest Model)")
st.markdown("**Kelompok 6 - Universitas Nusa Putra**")

# === Load Model & Encoder ===
@st.cache_resource
def load_model_and_encoders():
    with open("randomforest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_model_and_encoders()

# === Upload Dataset ===
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Unggah file CSV dataset HIV", type="csv")

def preprocess_input(df, label_encoders):
    df = df.copy()
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")
    st.write("Contoh data:", df.head())

    if 'Result' not in df.columns:
        st.error("❌ Kolom 'Result' tidak ditemukan. Pastikan dataset Anda memiliki kolom ini.")
        st.stop()

    df = df[df['Result'].isin([0, 1])]  # filter kelas biner
    X = df.drop('Result', axis=1)
    y_true = df['Result']

    try:
        X_encoded = preprocess_input(X, label_encoders)
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat encoding: {e}")
        st.stop()

    y_pred = model.predict(X_encoded)

    # Evaluasi
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.subheader("📊 Evaluasi Model")
    st.write(f"**Akurasi:** {acc:.2%}")
    st.write(f"**Precision:** {prec:.2%}")
    st.write(f"**Recall:** {rec:.2%}")
    st.write(f"**F1-Score:** {f1:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    # Hasil prediksi
    st.subheader("📥 Unduh Prediksi")
    df_out = df.copy()
    df_out['Prediksi'] = y_pred
    st.write(df_out.head())

    csv_out = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Prediksi", csv_out, file_name="hasil_prediksi.csv", mime="text/csv")
