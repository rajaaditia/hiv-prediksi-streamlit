import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
    st.write("Data setelah Encoding:")
    st.dataframe(data.head())

    X = data.drop('Result', axis=1)
    y = data['Result']

    # Split data 80% train, 20% test
    st.header("3. Split Data (80% Train, 20% Test)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.write(f"Jumlah Data Training: {X_train.shape[0]} | Jumlah Data Testing: {X_test.shape[0]}")

    # Training dan Hyperparameter Tuning
    st.header("4. Pelatihan Model dengan GridSearchCV")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluasi
    st.subheader("Evaluasi Model")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

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
    importances = best_model.feature_importances_ * 100
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
    st.download_button("Download Evaluasi", pd.DataFrame(report).transpose().to_csv(), file_name="evaluasi_model.csv")
