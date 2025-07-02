# eda_utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Загрузка и подготовка данных
df = pd.read_excel('Данные_для_курсовои_Классическое_МО.xlsx')
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

# Импьютация
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Масштабирование
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
