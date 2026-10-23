# KLASIFIKASI DATASET IRIS DENGAN DECISION TREE

# 1. IMPORT LIBRARY
import pandas as pd                                         # untuk membaca dan mengolah data tabel (CSV)
import matplotlib.pyplot as plt                             # untuk membuat grafik / visualisasi pohon
from sklearn.model_selection import train_test_split        # untuk membagi data latih dan data uji
from sklearn.tree import DecisionTreeClassifier, plot_tree  # model Decision Tree dan fungsi untuk menggambarnya
from sklearn.metrics import (                               # fungsi untuk mengukur kinerja model
    accuracy_score,                                         # persentase tebakan yang benar
    confusion_matrix,                                       # tabel tebakan benar/salah per kelas
    classification_report,                                  # ringkasan precision, recall, f1-score
)


# 2. MEMBACA DATA
df = pd.read_csv("Iris.csv")             # baca file Iris.csv menjadi DataFrame bernama df

print("5 baris pertama data:")
print(df.head())                         # tampilkan 5 baris awal untuk memastikan data terbaca


# 3. PRA-PEMROSESAN DATA
# Kolom Id hanya nomor urut, tidak berguna untuk klasifikasi,
# jadi dibuang agar tidak mengganggu model.
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Pisahkan fitur (input) dan label (target yang ingin diprediksi)
X = df.drop(columns=["Species"])         # X = fitur: panjang/lebar sepal dan petal
y = df["Species"]                        # y = label: jenis spesies bunga


# 4. MEMBAGI DATA LATIH DAN DATA UJI
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% data untuk pengujian, 80% untuk pelatihan
    random_state=42,     # angka acak tetap agar hasil selalu sama setiap dijalankan
    stratify=y,          # jaga proporsi tiap spesies tetap seimbang di data latih dan uji
)

print("\nJumlah data latih:", len(X_train))
print("Jumlah data uji  :", len(X_test))


# 5. MEMBUAT DAN MELATIH MODEL
model = DecisionTreeClassifier(
    criterion="gini",    # ukuran untuk memilih pemisahan terbaik (alternatif: "entropy")
    max_depth=3,         # batasi kedalaman pohon agar tidak overfitting dan mudah dibaca
    random_state=42,     # agar hasil pohon konsisten
)

model.fit(X_train, y_train)              # proses pelatihan: model belajar pola dari data latih


# 6. PREDIKSI DAN EVALUASI MODEL
y_pred = model.predict(X_test)           # model menebak spesies dari data uji

# Akurasi: berapa persen tebakan yang benar
print("\nAkurasi:", accuracy_score(y_test, y_pred))

# Confusion matrix: baris = label sebenarnya, kolom = hasil tebakan.
# Angka di diagonal adalah tebakan yang benar.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report: precision, recall, f1-score untuk tiap spesies
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7. FITUR PALING BERPENGARUH
# Menunjukkan seberapa besar kontribusi tiap fitur dalam membuat keputusan
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nPentingnya Fitur:")
print(importance.sort_values(ascending=False))


# 8. VISUALISASI POHON KEPUTUSAN
plt.figure(figsize=(14, 8))              # ukuran gambar (lebar, tinggi)
plot_tree(
    model,
    feature_names=X.columns,             # tampilkan nama fitur di setiap node
    class_names=model.classes_,          # tampilkan nama spesies di setiap daun
    filled=True,                         # beri warna sesuai kelas mayoritas
)
plt.title("Decision Tree - Klasifikasi Iris")
plt.show()                               # tampilkan gambar