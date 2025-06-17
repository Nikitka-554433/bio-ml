# main.py

from eda.exploratory_data_analysis import load_data, analyze_missing_values, plot_correlation_matrix
from regression.regression_models import get_regressors, get_param_grid, train_and_evaluate_models
from classification.binary_classification import (
    create_binary_targets,
    split_classification_data,
    get_classifiers,
    evaluate_classifiers,
    plot_roc_curve
)
from ensemble.stacking_ensemble import get_stacking_model, evaluate_stacking_model

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd

# === 1. Загрузка и первичный анализ ===
df = load_data('Данные_для_курсовои_Классическое_МО.xlsx')
analyze_missing_values(df)
plot_correlation_matrix(df)

# === 2. Предобработка ===
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

pca = PCA(n_components=0.95)
X = pca.fit_transform(df_scaled)

y_ic50 = df_scaled['IC50, mM'].values
y_cc50 = df_scaled['CC50, mM'].values
y_si = df_scaled['SI'].values

# === 3. Деление данных для регрессии ===
def split_targets(y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

regression_splits = {
    "IC50": split_targets(y_ic50),
    "CC50": split_targets(y_cc50),
    "SI": split_targets(y_si)
}

# === 4. Регрессия ===
regressors = get_regressors()
param_grid = get_param_grid()
regression_results = train_and_evaluate_models(regressors, param_grid, regression_splits)

print("\n=== Регрессия ===")
for model_name, res in regression_results.items():
    print(f"\n{model_name}:")
    for target, scores in res.items():
        print(f"  {target}: MSE = {scores['MSE']:.4f}, R² = {scores['R2']:.4f}")

# === 5. Stacking ===
stack_model = get_stacking_model()
stacking_results = evaluate_stacking_model(stack_model, regression_splits)

print("\n=== Stacking Ensemble ===")
for target, scores in stacking_results.items():
    print(f"  {target}: MSE = {scores['MSE']:.4f}, R² = {scores['R2']:.4f}")

# === 6. Классификация ===
y_dict = create_binary_targets(y_ic50, y_cc50, y_si)
classification_splits = split_classification_data(X, y_dict)
classifiers = get_classifiers()
classification_results = evaluate_classifiers(classifiers, classification_splits)

print("\n=== Классификация ===")
for clf_name, result in classification_results.items():
    print(f"\n{clf_name}:")
    for target, acc in result.items():
        print(f"  {target}: Accuracy = {acc:.4f}")

# === 7. ROC-кривая для задачи SI > 8 ===
clf_roc = classifiers["Gradient Boosting"]
X_train, X_test, y_train, y_test = classification_splits['SI>8']
clf_roc.fit(X_train, y_train)
plot_roc_curve(clf_roc, X_test, y_test)
