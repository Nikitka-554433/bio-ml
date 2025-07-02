# IC50
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from eda.eda_utils import df_scaled

# Бинаризация IC50
median_ic50 = df['IC50, mM'].median()
y_ic50_bin = (df['IC50, mM'] > median_ic50).astype(int)

# Признаки
X = df_scaled.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])

# Данные
X_train, X_test, y_train, y_test = train_test_split(X, y_ic50_bin, test_size=0.2, random_state=42)

# Модели с подбором параметров
models = {
    "LogisticRegression": GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {
        "C": [0.01, 0.1, 1, 10]
    }, cv=5),

    "RandomForest": GridSearchCV(RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    }, cv=5),

    "GradientBoosting": GridSearchCV(GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1]
    }, cv=5)
}

# Обучение и результаты
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, preds))
    print(f"  Best params: {model.best_params_}")