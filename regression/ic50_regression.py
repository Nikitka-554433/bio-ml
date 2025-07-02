# IC50
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from eda.eda_utils import df_scaled

# Целевая переменная
y_ic50 = df_scaled['IC50, mM']
X_ic50 = df_scaled.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])  # Убираем таргеты
X_train, X_test, y_train, y_test = train_test_split(X_ic50, y_ic50, test_size=0.2, random_state=42)

# Модели и гиперпараметры
models = {
    "Linear": LinearRegression(),
    "Ridge": GridSearchCV(Ridge(), param_grid={"alpha": [0.01, 0.1, 1, 10]}, cv=5),
    "Lasso": GridSearchCV(Lasso(), param_grid={"alpha": [0.001, 0.01, 0.1, 1]}, cv=5),
    "GradientBoosting": GridSearchCV(GradientBoostingRegressor(), param_grid={
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1]
    }, cv=5)
}

# Оценка
results_ic50 = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_ic50[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }
    print(f"{name}: MSE = {results_ic50[name]['MSE']:.4f}, R² = {results_ic50[name]['R²']:.4f}")

# Сравнение
import matplotlib.pyplot as plt

model_names = list(results_ic50.keys())
r2_scores = [results_ic50[m]["R²"] for m in model_names]

plt.figure(figsize=(8, 4))
sns.barplot(x=model_names, y=r2_scores, palette="viridis")
plt.title("Сравнение моделей по R² (регрессия IC50)")
plt.ylabel("R²")
plt.grid(True)
plt.show()

# Оценка и вывод параметров
results_ic50 = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_ic50[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }

    print(f"{name}: MSE = {results_ic50[name]['MSE']:.4f}, R² = {results_ic50[name]['R²']:.4f}")

    if isinstance(model, GridSearchCV):
        print(f"  Best params: {model.best_params_}")
# Значения
r2_ic50 = {
    'Linear': 0.2316,
    'Ridge': 0.3731,
    'Lasso': 0.3100,
    'GradientBoosting': 0.4225
}

# Визуализация IC50
plt.figure(figsize=(8, 4))
sns.barplot(x=list(r2_ic50.keys()), y=list(r2_ic50.values()), palette="Greens_d")
plt.title("Сравнение моделей по R² (регрессия IC50)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(True)
plt.show()