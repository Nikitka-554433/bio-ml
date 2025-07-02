import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from eda.eda_utils import df_scaled

# Выбираем порог SI > 4
df_scaled['SI_gt_4'] = (df_scaled['SI'] > 4).astype(int)

# Признаки и целевая переменная
X = df_scaled.drop(columns=['IC50, mM', 'CC50, mM', 'SI', 'SI_gt_4'])
y = df_scaled['SI_gt_4']

# Разбивка данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE с уменьшенным числом соседей
sm = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# Настройка моделей
models = {
    "LogisticRegression": GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid={"C": [0.01, 0.1, 1, 10]},
        cv=5,
        scoring='f1'
    ),
    "RandomForest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        cv=5,
        scoring='f1'
    ),
    "GradientBoosting": GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        cv=5,
        scoring='f1'
    )
}

# Обучение моделей
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Best params: {model.best_params_}\n")