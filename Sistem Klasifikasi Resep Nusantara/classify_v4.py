import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.combine import SMOTEENN
from collections import Counter, defaultdict

# === 1. Load dan Persiapan Data ===
df = pd.read_csv("Resep Masakan Indonesia.csv")  # <- disesuaikan
df = df.dropna(subset=['BAHAN'])  # Hapus data kosong
df['NAMA DAERAH'] = df['NAMA DAERAH'].str.strip()

# === 2. TF-IDF dan Label Encoding ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['BAHAN'])
le = LabelEncoder()
y = le.fit_transform(df['NAMA DAERAH'])

# === 3. Balancing dengan SMOTEENN ===
sampler = SMOTEENN(random_state=42)
X_bal, y_bal = sampler.fit_resample(X, y)

# === 4. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.25, stratify=y_bal, random_state=42
)

# === 5. Inisialisasi dan Latih Model ===
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='cosine')
model.fit(X_train, y_train)

# === 6. Evaluasi Akurasi ===
y_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_bal, y_bal, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))

print(f"\n📊 Akurasi keseluruhan model di data test: {test_acc:.2%}")
print(f"🎯 Train Accuracy: {train_acc:.2%}")
print(f"🎯 Test Accuracy : {test_acc:.2%}")
print(f"🔁 10-Fold Cross-validated accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# === 7. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === 8. Fungsi Klasifikasi Interaktif ===
def klasifikasi_bahan(input_bahan):
    input_bahan = input_bahan.lower()
    input_vect = vectorizer.transform([input_bahan])

    pred_label = model.predict(input_vect)
    pred_daerah = le.inverse_transform(pred_label)[0]

    probas = model.predict_proba(input_vect)
    confidence = probas.max() * 100

    similarities = cosine_similarity(input_vect, X).flatten()
    idx_terdekat = similarities.argmax()
    judul_masakan_terdekat = df.iloc[idx_terdekat]['JUDUL MASAKAN']

    return judul_masakan_terdekat, pred_daerah, confidence

def parse_bahan(bahan_str):
    bahan_str = str(bahan_str).strip("[]()")
    return [item.strip().lower() for item in bahan_str.split(",") if item.strip()]

df['bahan_list'] = df['BAHAN'].apply(parse_bahan)
all_bahan = [b for sublist in df['bahan_list'] for b in sublist]
top_bahan = Counter(all_bahan).most_common(10)

print("\n🔥 Top 10 Bahan yang Paling Sering Digunakan:")
for i, (bahan, jumlah) in enumerate(top_bahan, 1):
    print(f"{i}. {bahan} ({jumlah} resep)")

# === Hitung bahan per daerah ===
daerah_bahan = defaultdict(list)
for _, row in df.iterrows():
    daerah = row['NAMA DAERAH'].strip().lower()
    daerah_bahan[daerah].extend(row['bahan_list'])

top_bahan_df = pd.DataFrame(top_bahan, columns=['Bahan', 'Jumlah'])

plt.figure(figsize=(10, 5))
sns.barplot(data=top_bahan_df, x='Jumlah', y='Bahan', palette='viridis')
plt.title("Top 10 Bahan yang Paling Sering Digunakan")
plt.xlabel("Jumlah Kemunculan")
plt.ylabel("Nama Bahan")
plt.tight_layout()
plt.show()

# === Bahan unik per daerah ===
daerah_bahan_freq = {d: Counter(bahan) for d, bahan in daerah_bahan.items()}

# Pemetaan bahan → set daerah di mana bahan itu muncul
semua_bahan_daerah = defaultdict(set)
for daerah, bahan_freq in daerah_bahan_freq.items():
    for b in bahan_freq:
        semua_bahan_daerah[b].add(daerah)

# Bahan yang hanya muncul di satu daerah
bahan_unik_per_daerah = defaultdict(list)
for bahan, daerah_set in semua_bahan_daerah.items():
    if len(daerah_set) == 1:
        satu_daerah = list(daerah_set)[0]
        bahan_unik_per_daerah[satu_daerah].append(bahan)

# === Tampilkan bahan unik per daerah ===
print("\n🌿 Bahan Unik per Daerah (max 10/buah):")
for daerah, bahan2 in bahan_unik_per_daerah.items():
    print(f"\n📍 {daerah.capitalize()} (jumlah unik: {len(bahan2)}):")
    for b in sorted(bahan2):  # maksimal 10 ditampilkan
        print(f"   - {b}")

# === 9. CLI Interaktif ===
print("\n🍽️ Sistem Klasifikasi Daerah & Masakan Berdasarkan Bahan")
print("Ketik 'exit' untuk keluar.")
while True:
    bahan_input = input("\nMasukkan bahan makanan: ")
    if bahan_input.lower() == 'exit':
        print("👋 Terima kasih sudah menggunakan sistem klasifikasi!")
        break
    judul_masakan, daerah, conf = klasifikasi_bahan(bahan_input)
    print(f"🍲 Nama masakan terdekat: {judul_masakan}")
    print(f"📍 Asal daerah: {daerah} (Confidence: {conf:.2f}%)")
