# EDA
# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, shapiro
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Стиль графиков
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Загрузка данных
df = pd.read_excel('Данные_для_курсовои_Классическое_МО.xlsx')
print("Размер датасета:", df.shape)
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

# Импьютация пропусков
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Масштабирование
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

# Первичный осмотр
print("\nПропущенные значения:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nСтатистики по целевым переменным:")
targets = ['IC50, mM', 'CC50, mM', 'SI']
print(df[targets].describe())

# Распределения целевых переменных
for col in targets:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Распределение {col}')
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.grid(True)
    plt.show()

# Boxplots + выбросы
for col in targets:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Выбросы в {col}')
    plt.show()

# Оценка нормальности
for col in targets:
    stat, p = shapiro(df[col].dropna())
    print(f'Shapiro-Wilk test для {col}: p-value = {p:.4f} → {"нормально" if p > 0.05 else "не нормально"}')

# Корреляция признаков с таргетами
corr_matrix = df.corr(numeric_only=True)

for target in targets:
    corr_target = corr_matrix[target].drop(target).sort_values(ascending=False)
    top_corr = corr_target.head(10)
    print(f"\n10 наиболее коррелирующих признаков с {target}:")
    print(top_corr)

    plt.figure()
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f"Сильнейшие корреляции с {target}")
    plt.xlabel("Коэффициент корреляции")
    plt.grid(True)
    plt.show()

# Пропущенные значения — обработка
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Отделяем X и y
X = df_imputed.drop(columns=targets)
y_ic50 = df_imputed['IC50, mM']
y_cc50 = df_imputed['CC50, mM']
y_si = df_imputed['SI']

# Масштабирование
scaler_std = StandardScaler()
scaler_rob = RobustScaler()

X_scaled_std = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns)
X_scaled_rob = pd.DataFrame(scaler_rob.fit_transform(X), columns=X.columns)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled_std)
print("\nPCA: объяснённая дисперсия (95%):", sum(pca.explained_variance_ratio_))

plt.figure()
sns.lineplot(x=range(1, len(pca.explained_variance_ratio_)+1),
             y=np.cumsum(pca.explained_variance_ratio_))
plt.title("Суммарная объяснённая дисперсия PCA")
plt.xlabel("Число компонент")
plt.ylabel("Объяснённая дисперсия")
plt.grid(True)
plt.show()