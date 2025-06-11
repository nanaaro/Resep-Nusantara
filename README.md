# Resep-Nusantara
This project builds a classification system for traditional Indonesian recipes based on their ingredients, predicting the **region of origin** using text mining, preprocessing, and machine learning techniques.

---

# Objective
Predict the regional origin of Indonesian dishes based on their list of ingredients using **K-Nearest Neighbors (KNN)** classifier enhanced with **TF-IDF vectorization**, **SMOTEENN balancing**, and domain-specific **ingredient mapping**.

---

# Tech Stack
- Python
- Pandas, NumPy
- scikit-learn, imbalanced-learn
- TF-IDF (text vectorization)
- KNN Classifier
- Matplotlib, Seaborn

---

# Preprocessing
- Cleaning, lowercasing, and removing special characters
- Stopword removal specific to cooking (e.g., "rebus", "tumis", "secukupnya")
- Mapping ingredient variants (e.g., *cabe rawit*, *lombok rawit*) into normalized categories

# Ingredient Categorization
- Uses custom mapping (`Padanan Bahan.xlsx`) to group similar terms under categories like `cabai`, `ikan`, `minyak`, etc.

# Classification
- TF-IDF vectorization of cleaned ingredients
- KNN classifier with cosine similarity
- Balancing using **SMOTEENN**
- Evaluation via accuracy, confusion matrix, and cross-validation

# Interactive Mode (CLI)
- Users can input ingredients to receive predicted region & suggested dish

---

# Evaluation Results
- Accuracy: ~87% (test set)
- 10-fold cross-validation: ~86%
- Includes classification report and confusion matrix

---

# How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run classification:
   python classify_v4.py

