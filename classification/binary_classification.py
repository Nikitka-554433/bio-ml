# classification/binary_classification.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def create_binary_targets(y_ic50, y_cc50, y_si):
    import numpy as np
    median_ic50 = np.median(y_ic50)
    median_cc50 = np.median(y_cc50)
    median_si = np.median(y_si)

    y_class_ic50 = (y_ic50 > median_ic50).astype(int)
    y_class_cc50 = (y_cc50 > median_cc50).astype(int)
    y_class_si = (y_si > median_si).astype(int)
    y_class_si_8 = (y_si > 8).astype(int)

    return {
        'IC50': y_class_ic50,
        'CC50': y_class_cc50,
        'SI': y_class_si,
        'SI>8': y_class_si_8
    }

def split_classification_data(X, y_dict):
    splits = {}
    for key, y in y_dict.items():
        splits[key] = train_test_split(X, y, test_size=0.2, random_state=42)
    return splits

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier()
    }

def evaluate_classifiers(classifiers, splits):
    results = {}
    for name, clf in classifiers.items():
        res = {}
        for target, (X_train, X_test, y_train, y_test) in splits.items():
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            res[target] = acc
        results[name] = res
    return results

def plot_roc_curve(clf, X_test, y_test):
    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая: SI > 8 (Gradient Boosting)")
    plt.legend()
    plt.grid(True)
    plt.show()

