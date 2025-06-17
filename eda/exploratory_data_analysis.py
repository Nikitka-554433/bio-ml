# eda/exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_excel(filepath)
    print("Размер данных:", df.shape)
    print("Пример записей:")
    print(df.head())
    return df

def analyze_missing_values(df):
    missing = df.isnull().sum()
    print("\nПризнаки с пропусками:\n", missing[missing > 0])

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица исходных признаков')
    plt.show()
