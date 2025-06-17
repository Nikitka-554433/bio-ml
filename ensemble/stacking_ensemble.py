# ensemble/stacking_ensemble.py

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_stacking_model():
    base_models = [
        ('ridge', Ridge(alpha=1)),
        ('lasso', Ridge(alpha=0.1)),  # Можно заменить на Lasso, но для совместимости
        ('gbr', StackingRegressor)    # Но тут лучше GradientBoostingRegressor
    ]
    # Исправим на корректный базовый ансамбль
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    base_models = [
        ('ridge', Ridge(alpha=1)),
        ('lasso', Lasso(alpha=0.1)),
        ('gbr', GradientBoostingRegressor())
    ]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
    return stack

def evaluate_stacking_model(model, splits):
    results = {}
    for target, (X_train, X_test, y_train, y_test) in splits.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[target] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
    return results

