import pandas as pd
import re

# 1. Baca dataset mentah
df = pd.read_excel("Preprocessing.xlsx")

# 2. Baca file padanan kata
padanan_df = pd.read_excel("Padanan Bahan.xlsx")  # sesuaikan nama file kamu
padanan_df.dropna(inplace=True)

# 3. Kamus singkatan/typo satuan
singkatan_dict = {
    "g": "gram", "gr": "gram", "sdm": "sendok makan", "sdt": "sendok teh",
    "btr": "butir", "bh": "buah", "ml": "mililiter", "cm": "sentimeter",
    "bwg": "bawang", "btg": "batang", "lbr": "lembar", "lg": "lagi",
    "dg": "dengan", "dgn": "dengan"
}

# 4. Stopwords umum dapur
stopwords_resep = set([
    "dan", "dengan", "serta", "hingga", "lalu", "kemudian", "sampai", "agar", "untuk",
    "dari", "pada", "adalah", "jika", "setelah", "dalam", "itu", "tersebut", "yang",
    "ke", "di", "sebagai", "oleh", "karena", "akan", "atau", "jadi", "tiap",
    "sebelum", "selama", "tanpa", "saat", "gram", "makan", "buah", "siung",
    "aduk", "matang", "tumis", "angkat", "tambahkan", "masukkan", "potong", "haluskan",
    "campur", "bagi", "ambil", "rebus", "masak", "bakar", "goreng", "cuci", "simpan",
    "tunggu", "dinginkan", "kecilkan", "rata", "sedok", "iris", "secukupnya", "lembar",
    "butir", "sentimeter", "mililiter", "memarkan", "besar", "sajikan", "sajikanlah",
    "panaskan", "halus", "keriting", "kriting", "kecil", "besar", "sedikit", "banyak",
    "dadu", "tipis", "cc", "liter", "pekat", "sedang", "bumbu"
])

# 5. Fungsi untuk membersihkan dan mengkategorikan bahan dari teks
def bersihkan_dan_kategorikan(text):
    if pd.isnull(text):
        return ""

    # Ganti singkatan
    for singkat, lengkap in singkatan_dict.items():
        pattern = r'\b' + re.escape(singkat) + r'\b'
        text = re.sub(pattern, lengkap, text)

    # Hapus karakter non-huruf dan ubah ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())

    # Tokenisasi dan hapus stopwords
    tokens = [t for t in text.split() if t not in stopwords_resep]
    text_clean = ' '.join(tokens)

    # Cek keberadaan padanan bahan di teks
    hasil_kategori = set()
    for _, row in padanan_df.iterrows():
        padanan = row['BAHAN'].strip().lower()
        kategori = row['KATEGORI'].strip()

        # Pencocokan kata persis
        pattern = r'\b' + re.escape(padanan) + r'\b'
        if re.search(pattern, text_clean):
            hasil_kategori.add(kategori)

    return '(' + ', '.join(sorted(hasil_kategori)) + ')' if hasil_kategori else ""

# 6. Aplikasikan ke kolom BAHAN
df['BAHAN'] = df['BAHAN'].astype(str).apply(bersihkan_dan_kategorikan)

# 7. Simpan hasil
df.to_excel("Resep Masakan Indonesia.xlsx", index=False)
