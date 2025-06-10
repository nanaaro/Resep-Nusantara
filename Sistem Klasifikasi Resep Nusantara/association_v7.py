import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint

# 1. Load data dari CSV
df = pd.read_csv("Resep Masakan Indonesia.csv")

# 2. Bersihkan kolom target
df['NAMA DAERAH'] = df['NAMA DAERAH'].str.strip().str.lower()

# 3. Konversi kolom BAHAN (dalam format string list atau dipisah koma) ke list
def parse_bahan(bahan_str):
    bahan_str = str(bahan_str).strip("[]()")
    return [item.strip().lower() for item in bahan_str.split(",") if item.strip()]

df['bahan_list'] = df['BAHAN'].apply(parse_bahan)

# 4. Gabungkan bahan menjadi string (untuk TF-IDF input)
df['bahan_tfidf'] = df['bahan_list'].apply(lambda x: ' '.join(x))

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['bahan_tfidf'])

# 6. Encode target
le = LabelEncoder()
y = le.fit_transform(df['NAMA DAERAH'])


# 7. Fungsi evaluasi model
def evaluate_model(X, y, balancing_method, split_ratio, search_type):
    print(f"\n🔎 Evaluasi: Split={int(split_ratio*100)}%, Balancing={balancing_method}, Search={search_type}")

    if balancing_method == 'smote':
        balancer = SMOTE(random_state=42)
    elif balancing_method == 'undersample':
        balancer = RandomUnderSampler(random_state=42)
    elif balancing_method == 'randomoversample':
        from imblearn.over_sampling import RandomOverSampler
        balancer = RandomOverSampler(random_state=42)
    elif balancing_method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        balancer = ADASYN(random_state=42)
    elif balancing_method == 'borderline_smote':
        from imblearn.over_sampling import BorderlineSMOTE
        balancer = BorderlineSMOTE(random_state=42)
    elif balancing_method == 'tomek_links':
        from imblearn.under_sampling import TomekLinks
        balancer = TomekLinks()
    elif balancing_method == 'enn':
        from imblearn.under_sampling import EditedNearestNeighbours
        balancer = EditedNearestNeighbours()
    elif balancing_method == 'nearmiss':
        from imblearn.under_sampling import NearMiss
        balancer = NearMiss()
    elif balancing_method == 'smote_tomek':
        from imblearn.combine import SMOTETomek
        balancer = SMOTETomek(random_state=42)
    elif balancing_method == 'smote_enn':
        from imblearn.combine import SMOTEENN
        balancer = SMOTEENN(random_state=42)
    else:
        balancer = None

    if balancer:
        X_bal, y_bal = balancer.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=split_ratio, random_state=42, stratify=y_bal
    )

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine', 'manhattan', 'minkowski', 'hamming', 'jaccard', 'chebyshev']
    }

    if search_type == 'grid':
        search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
    else:
        search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_iter=6, random_state=42, n_jobs=-1)

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"📌 Best Params: {search.best_params_}")
    print(f"📊 Akurasi: {accuracy:.2%}")

    return {
        'split_ratio': f"{int((1-split_ratio)*100)}:{int(split_ratio*100)}",
        'balancing': balancing_method,
        'search': search_type,
        'accuracy': accuracy,
        'best_params': search.best_params_,
        'model': best_model
    }

# 8. Evaluasi kombinasi split, balancing, dan search
results = []
split_ratios = [0.2, 0.25, 0.3]
balancing_methods = ['smote',
    'adasyn',
    'borderline_smote',
    'random_undersample',
    'tomek_links',
    'enn',
    'nearmiss',
    'smote_tomek',
    'smote_enn',
    'smote',             
    'undersample',        
    'randomoversample']
search_types = ['grid',
    'random',
    'bayesian',
    'halving_grid',
    'genetic']

for split in split_ratios:
    for balance in balancing_methods:
        for search in search_types:
            res = evaluate_model(X, y, balance, split, search)
            results.append(res)

# 9. Tampilkan hasil terbaik
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='accuracy', ascending=False)

# Konversi ke DataFrame
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='accuracy', ascending=False)

# Cetak 5 hasil terbaik termasuk best_params
print("\n🏆 5 Kombinasi Terbaik:")
for idx, row in df_results_sorted.head(5).iterrows():
    try:
        split_display = f"{int(float(row['split_ratio'])*100)}%"
    except:
        split_display = f"{row['split_ratio']} (Invalid)"
    print(f"{idx+1}. Split={split_display}, Balancing={row['balancing']}, Search={row['search']}, "
          f"Akurasi={row['accuracy']*100:.2f}%\n   Best Params: {row['best_params']}")

# 10. Visualisasi akurasi
plt.figure(figsize=(12, 6))
sns.barplot(data=df_results_sorted, x='split_ratio', y='accuracy', hue='balancing')
plt.title("Perbandingan Akurasi KNN Berdasarkan Teknik Balancing dan Split")
plt.ylim(0.5, 1.0)
plt.ylabel("Akurasi")
plt.xlabel("Rasio Data Train:Test")
plt.tight_layout()
plt.show()
