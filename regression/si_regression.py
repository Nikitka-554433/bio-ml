# SI
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Целевая переменная SI
y_si = df_scaled['SI']
X_si = df_scaled.drop(columns=['IC50, mM', 'CC50, mM', 'SI'])

X_train, X_test, y_train, y_test = train_test_split(X_si, y_si, test_size=0.2, random_state=42)

# Модели с GridSearch
models = {
    'Linear': LinearRegression(),
    'Ridge': GridSearchCV(Ridge(), param_grid={'alpha': [0.01, 0.1, 1, 10]}, cv=5),
    'Lasso': GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': [0.001, 0.01, 0.1, 1]}, cv=5),
    'GradientBoosting': GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid={
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }, cv=5)
}

results_si = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results_si[name] = (mse, r2)

    print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")
    if isinstance(model, GridSearchCV):
        print(f"  Best params: {model.best_params_}")