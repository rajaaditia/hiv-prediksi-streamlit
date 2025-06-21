import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

st.set_page_config(page_title="Prediksi HIV", layout="wide")

# Judul Aplikasi
st.title("Prediksi Risiko HIV dengan Random Forest")
st.markdown("**Kelompok 6 - Universitas Nusa Putra**")

# Upload dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Unggah file CSV dataset HIV", type="csv")

def load_and_preprocess(data):
    df = data.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat!")
    st.write("Contoh Data:", df.head())

    # Preprocessing
    st.header("2. Preprocessing")
    data = load_and_preprocess(df)

    # Filter hanya kelas 0 dan 1 (binary classification)
    data = data[data['Result'].isin([0, 1])].reset_index(drop=True)
    st.write("Data setelah Encoding dan Filtering (hanya kelas 0 & 1):")
    st.dataframe(data.head())

    # Sampling hanya 20% dari data untuk meniru laporan Colab
    data_sampled = data.sample(frac=0.4, random_state=42).reset_index(drop=True)

    # SPLIT DATA SEBELUM BALANCING
    X = data_sampled.drop('Result', axis=1)
    y = data_sampled['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)

    # BALANCING HANYA DATA TRAINING
    train_df = pd.concat([X_train, y_train], axis=1)
    positive = train_df[train_df['Result'] == 1]
    negative = train_df[train_df['Result'] == 0]

    if len(positive) > 0:
        positive_upsampled = resample(positive,
                                      replace=True,
                                      n_samples=len(negative),
                                      random_state=42)
        balanced_train = pd.concat([negative, positive_upsampled]).sample(frac=1, random_state=42)
        X_train = balanced_train.drop('Result', axis=1)
        y_train = balanced_train['Result']
    else:
        st.error("Jumlah data kelas 1 terlalu sedikit setelah sampling. Silakan gunakan lebih banyak data atau ubah sampling.")
        st.stop()

    # Verifikasi balancing
    st.write("Distribusi data training setelah balancing:", y_train.value_counts())
    st.write("Distribusi kelas di data testing:", y_test.value_counts())
    st.write(f"Jumlah Data Training: {X_train.shape[0]} | Jumlah Data Testing: {X_test.shape[0]}")

    # Train model tanpa tuning (seperti di Colab)
    st.header("3. Pelatihan Model Random Forest Tanpa Tuning")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi manual (binary classification)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)

    st.subheader("Evaluasi Model")
    st.markdown(f"**Akurasi:** {acc * 100:.2f}%")
    st.markdown(f"**Precision:** {prec * 100:.2f}%")
    st.markdown(f"**Recall:** {rec * 100:.2f}%")
    st.markdown(f"**F1-Score:** {f1 * 100:.2f}%")

    # Distribusi hasil prediksi
    st.write("Distribusi prediksi model:", pd.Series(y_pred).value_counts())

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    importances = model.feature_importances_ * 100
    feature_df = pd.DataFrame({'Fitur': X.columns, 'Importance (%)': importances})
    feature_df = feature_df.sort_values(by='Importance (%)', ascending=True)

    fig2, ax2 = plt.subplots()
    sns.barplot(data=feature_df, x='Importance (%)', y='Fitur', palette='viridis', ax=ax2)
    for patch in ax2.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax2.text(width + 1, y, f'{width:.2f}%', va='center')
    ax2.set_title("Pengaruh Fitur terhadap Prediksi")
    st.pyplot(fig2)

    # Download hasil
    st.subheader("Unduh Laporan Evaluasi")
    eval_df = pd.DataFrame({
        'Metrik': ['Akurasi', 'Precision', 'Recall', 'F1-score'],
        'Hasil (%)': [f"{acc * 100:.2f}%", f"{prec * 100:.2f}%", f"{rec * 100:.2f}%", f"{f1 * 100:.2f}%"]
    })
    st.dataframe(eval_df)
    st.download_button("Download Evaluasi", eval_df.to_csv(index=False), file_name="evaluasi_model.csv")
