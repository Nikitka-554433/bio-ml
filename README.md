# 🧪 Анализ биологических данных (IC50, CC50, SI)

Необходимо построить прогноз, позволяющий подобрать наиболее эффективное сочетание параметров для создания лекарственных препаратов.

## 📁 Структура проекта

bioactivity_ml
├── eda
| |__eda_utils.py # Данные
│ └── eda_analys.py # Исследовательский анализ 
данных
│
├── regression
│ ├── ic50_regression.py # Регрессия для IC50
│ ├── cc50_regression.py # Регрессия для CC50
│ └── si_regression.py # Регрессия для SI
│
├── classification
│ ├── ic50_classif.py # Классификация IC50
│ ├── cc50_classif.py # Классификация CC50
│ └── si_classif.py/si_gt8_classif.py # Классификация SI и для SI_GT8
│ 
├── classif_experiments
│ └── si_gt4_smote_classification.py # Эксперимент с SMOTE (SI > 4)
│
├── reports/ # графики, метрики, выводы
│
├── .gitignore # Игнорирование лишних файлов
├── requirements.txt # Список зависимостей
└── Данные_для_курсовои.xlsx # Датасет (в .gitignore)

## ⚙️ Как запустить

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
2. EDA (исследовательский анализ)
bash

python eda/eda_analysis.py
3. Регрессия
bash

python regression/ic50_regression.py
python regression/cc50_regression.py
python regression/si_regression.py
4. Классификация
bash

python classification/ic50_classification.py
python classification/cc50_classification.py
python classification/si_classification.py
5. Эксперименты
bash

python classification_experiments/si_gt4_smote_classification.py
📦 Зависимости
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn (для SMOTE)

Устанавливаются через requirements.txt.


Визуализации и метрики можно в папке reports.

👤 Автор
Бодак Н. И.