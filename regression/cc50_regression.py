# CC50
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Целевая переменная
y_cc50 = df_scaled['CC50, mM']
X_cc50 = df_scaled.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])  # Убираем таргеты

# train/test
X_train, X_test, y_train, y_test = train_test_split(X_cc50, y_cc50, test_size=0.2, random_state=42)

# Модели с подбором гиперпараметров
models = {
    'Linear': LinearRegression(),
    'Ridge': GridSearchCV(Ridge(), param_grid={'alpha': [0.01, 0.1, 1, 10]}, cv=5),
    'Lasso': GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': [0.001, 0.01, 0.1, 1]}, cv=5),
    'GradientBoosting': GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid={
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }, cv=5)
}

# Обучение и оценка
mse_scores = {}
r2_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mse_scores[name] = mse
    r2_scores[name] = r2

    print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")
    if isinstance(model, GridSearchCV):
        print(f"  Best params: {model.best_params_}")
import seaborn as sns
import matplotlib.pyplot as plt

# Значения R² 
r2_cc50 = {
    'Linear': 0.3476,
    'Ridge': 0.4864,
    'Lasso': 0.4667,
    'GradientBoosting': 0.5883
}

r2_si = {
    'Linear': 0.76,
    'Ridge': 0.77,
    'Lasso': 0.75,
    'GradientBoosting': 0.79
}

# Визуализация CC50
plt.figure(figsize=(8, 4))
sns.barplot(x=list(r2_cc50.keys()), y=list(r2_cc50.values()), palette="Blues_d")
plt.title("Сравнение моделей по R² (регрессия CC50)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Визуализация SI
plt.figure(figsize=(8, 4))
sns.barplot(x=list(r2_si.keys()), y=list(r2_si.values()), palette="Purples_d")
plt.title("Сравнение моделей по R² (регрессия SI)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(True)
plt.show()