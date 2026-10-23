import pandas as pd

df = pd.read_csv("Iris.csv")

print(df.head())      # 5 baris pertama
print(df.info())      # tipe data dan jumlah baris
print(df.describe())  # statistik ringkas