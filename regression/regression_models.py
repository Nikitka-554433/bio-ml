# regression/regression_models.py

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def get_regressors():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

def get_param_grid():
    return {
        "Ridge": {"alpha": [0.1, 1, 10]},
        "Lasso": {"alpha": [0.01, 0.1, 1]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
    }

def evaluate_regression(model, splits):
    results = {}
    for target, (X_train, X_test, y_train, y_test) in splits.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[target] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
    return results

def train_and_evaluate_models(regressors, param_grid, splits):
    results = {}
    for name, model in regressors.items():
        if name in param_grid:
            # Для упрощения возьмём первый target для поиска гиперпараметров (IC50)
            grid = GridSearchCV(model, param_grid[name], cv=5)
            X_train, _, y_train, _ = splits['IC50']
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = model
        res = evaluate_regression(best_model, splits)
        results[name] = res
    return results
