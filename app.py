import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ------------------- BAGIAN 1: LOAD DATA ------------------- #
st.title("Prediksi Risiko HIV dengan Random Forest")

@st.cache_data
def load_data():
    return pd.read_csv("hIV_dataset.csv")

df = load_data()
st.subheader("Dataset HIV")
st.dataframe(df.head())

# ------------------- BAGIAN 2: CEK DISTRIBUSI KELAS ------------------- #
st.subheader("Distribusi Target (HIV result)")
st.bar_chart(df['HIV result'].value_counts())

# ------------------- BAGIAN 3: PREPROCESSING ------------------- #
st.subheader("Preprocessing Data")
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

st.write("Data setelah Label Encoding:")
st.dataframe(df_clean.head())

# ------------------- BAGIAN 4: SPLIT ------------------- #
X = df_clean.drop('HIV result', axis=1)
y = df_clean['HIV result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- BAGIAN 5: SMOTE ------------------- #
st.subheader("Balancing Data dengan SMOTE")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
st.write(f"Jumlah data setelah SMOTE: {X_resampled.shape[0]} sample")

# ------------------- BAGIAN 6: TRAINING MODEL ------------------- #
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# ------------------- BAGIAN 7: EVALUASI ------------------- #
st.subheader("Evaluasi Model")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ------------------- BAGIAN 8: FEATURE IMPORTANCE ------------------- #
st.subheader("Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x=importances, y=importances.index, ax=ax2)
ax2.set_title("Fitur yang Paling Mempengaruhi Prediksi HIV")
st.pyplot(fig2)

# ------------------- BAGIAN 9: PREDIKSI USER ------------------- #
st.subheader("Prediksi Risiko HIV pada Individu Baru")

def user_input_features():
    age = st.selectbox("Usia", df["Age"].unique())
    marital = st.selectbox("Status Pernikahan", df["Marital Status"].unique())
    std = st.selectbox("Riwayat STD", df["STD"].unique())
    education = st.selectbox("Latar Belakang Pendidikan", df["Educational Background"].unique())
    hiv_test = st.selectbox("Tes HIV Sebelumnya", df["HIV Test in Past Years"].unique())
    aids_ed = st.selectbox("Edukasi AIDS", df["AIDS Education"].unique())
    place = st.selectbox("Tempat Mencari Pasangan", df["Places of Seeking Sex Partners"].unique())
    orientation = st.selectbox("Orientasi Seksual", df["Sexual Orientation"].unique())
    drug = st.selectbox("Penggunaan Narkoba", df["Drug Taking"].unique())

    data = {
        "Age": age,
        "Marital Status": marital,
        "STD": std,
        "Educational Background": education,
        "HIV Test in Past Years": hiv_test,
        "AIDS Education": aids_ed,
        "Places of Seeking Sex Partners": place,
        "Sexual Orientation": orientation,
        "Drug Taking": drug
    }
    return pd.DataFrame([data])

input_df = user_input_features()
input_encoded = input_df.copy()

# Encoding input user
for col in input_encoded.columns:
    le = LabelEncoder()
    le.fit(df[col])
    input_encoded[col] = le.transform(input_df[col])

# Prediksi
if st.button("Prediksi"):
    hasil = model.predict(input_encoded)
    st.success(f"Hasil Prediksi: {'POSITIF HIV' if hasil[0]==1 else 'NEGATIF HIV'}")
