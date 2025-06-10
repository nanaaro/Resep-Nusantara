import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from PIL import Image

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Resep Masakan Indonesia.csv")
    df = df.dropna(subset=['BAHAN'])
    df['NAMA DAERAH'] = df['NAMA DAERAH'].str.strip()
    return df

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['BAHAN'])
    le = LabelEncoder()
    y = le.fit_transform(df['NAMA DAERAH'])~

    sampler = SMOTEENN()
    X_res, y_res = sampler.fit_resample(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_res, y_res)
    return model, vectorizer, le

# Klasifikasi input user + cari nama masakan terdekat
def klasifikasi_bahan(bahan_input, model, vectorizer, le, df):
    X_input = vectorizer.transform([bahan_input])
    neighbors = model.kneighbors(X_input, return_distance=False)
    pred_class = model.predict(X_input)[0]
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = np.mean(le.inverse_transform(model._y[neighbors[0]]) == pred_label) * 100

    # Cari nama masakan dari daerah hasil prediksi yang paling mirip
    df_pred_daerah = df[df['NAMA DAERAH'] == pred_label].copy()
    df_pred_daerah['similarity'] = df_pred_daerah['BAHAN'].apply(
        lambda x: vectorizer.transform([x]).dot(X_input.T).toarray()[0][0]
    )
    best_match = df_pred_daerah.sort_values(by='similarity', ascending=False).iloc[0]
    nama_masakan = best_match['JUDUL MASAKAN']

    return pred_label, confidence, nama_masakan

# Streamlit UI
st.title("🇮🇩 Klasifikasi Daerah Masakan Indonesia dengan KNN")
st.markdown("""
Model **K-Nearest Neighbors (KNN)** digunakan untuk mengidentifikasi **asal daerah masakan** berdasarkan **bahan yang digunakan**.
""")

# Load data dan latih model
df = load_data()
model, vectorizer, le = train_model(df)

# Input pengguna
st.subheader("🔍 Uji Coba Prediksi")
bahan_input = st.text_area("Masukkan daftar bahan makanan (dipisahkan dengan koma)",
                           "air, cabai merah, daging ayam, daun jeruk, garam, gula, santan")
if st.button("Prediksi"):
    daerah, conf, nama_masakan = klasifikasi_bahan(bahan_input, model, vectorizer, le, df)
    st.success(f"""
    📍 **Asal daerah:** {daerah}  
    🍛 **Nama masakan:** {nama_masakan}  
    🔍 **Confidence:** {conf:.2f}%
    """)

# Visualisasi akurasi model (statik)
st.subheader("📊 Hasil Evaluasi Model")
st.markdown("""
**Akurasi Model:**
- Akurasi pelatihan: 97.92%
- Akurasi pengujian: 95.32%
- 10-fold cross validation: 96.66% (std: 1.63%)

**Classification Report:**

| Daerah     | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Bali       | 0.95      | 0.97   | 0.96     | 112     |
| Jawa       | 0.99      | 0.92   | 0.95     | 89      |
| Kalimantan | 0.96      | 1.00   | 0.98     | 110     |
| Papua      | 0.94      | 1.00   | 0.97     | 123     |
| Sulawesi   | 0.95      | 0.93   | 0.94     | 111     |
| Sumatera   | 0.96      | 0.75   | 0.84     | 32      |

*Catatan: Sumatera memiliki jumlah data paling sedikit, sehingga recall-nya rendah.*
""")

# Confusion Matrix
st.subheader(" Confusion Matrix")
img = Image.open('confussion.jpg') 
st.image(img, use_container_width=True)



# Visualisasi bahan paling sering digunakan
st.subheader("🍽️ 10 Bahan Paling Sering Digunakan")

all_ingredients = df['BAHAN'].str.lower().str.split(', ')
all_flat = [item.strip() for sublist in all_ingredients.dropna() for item in sublist]
bahan_series = pd.Series(all_flat)
top10 = bahan_series.value_counts().head(10)

fig2, ax2 = plt.subplots()
sns.barplot(x=top10.values, y=top10.index, palette='viridis', ax=ax2)
ax2.set_xlabel('Frekuensi')
ax2.set_ylabel('Bahan')
ax2.set_title('Top 10 Bahan Paling Populer')
st.pyplot(fig2)
