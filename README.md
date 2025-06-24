# Прогноз одобрения кредита – ML-конвейер + FastAPI + деплой на AWS

Полный путь от «сырых» данных до боевого REST API, который классифицирует заявки на кредит как **Approved** или **Rejected** c точностью ≥ 95 %.

---

## 📌 Цели проекта

| Этап | Задача |
|------|--------|
| **EDA** | Детальный разведочный анализ (Exploratory Data Analysis) |
| **Моделирование** | Обучить и сравнить **5 алгоритмов** (LogReg, DT, RF, SVM, KNN) |
| **Качество** | Достичь ≥ 95 % accuracy (с учётом precision / recall) |
| **Сериализация** | Сохранить лучшую модель и пайплайн предобработки через `joblib` |
| **Сервис** | Поднять REST-endpoint на **FastAPI** для онлайн-предсказаний |
| **Деплой** | Развернуть сервис в AWS (EC2 или Elastic Beanstalk) |

---

## 🗂️ Описание данных

| Признак | Тип | Описание |
|---------|-----|----------|
| `loan_id` | int | Идентификатор заявки |
| `no_of_dependents` | int | Кол-во иждивенцев |
| `education` | cat | `Graduate / Not Graduate` |
| `self_employed` | cat | `Yes / No` |
| `income_annum` | float | Годовой доход (₹) |
| `loan_amount` | float | Запрашиваемая сумма (₹) |
| `loan_term` | int | Срок кредита (мес.) |
| `cibil_score` | int | Кредитный рейтинг (CIBIL) |
| `residential_assets_value` | float | Жилая недвижимость |
| `commercial_assets_value` | float | Коммерческая недвижимость |
| `luxury_assets_value` | float | Предметы роскоши |
| `bank_asset_value` | float | Активы в банке |
| `loan_status` | cat | **Целевая** `Approved / Rejected` |

---

## 🏗️ Общая схема

1. **EDA-ноутбук** – статистика и визуализации (гистограммы, heatmap, pairplot).  
2. **Предобработка**  
   * Заполнение пропусков  
   * `OneHotEncoder / LabelEncoder`  
   * `StandardScaler` для числовых столбцов  
3. **Обучение** (`train.py`)  
   * Сплит 80 / 20, кросс-валидация, при нужде `GridSearchCV`  
   * Метрики: Accuracy • Precision • Recall • F1 • ROC AUC  
4. **Выбор модели** – сравнение метрик + ROC-кривые → лучшая модель.  
5. **Сохранение** – модель + скейлер → `models/` (`joblib`).  
6. **FastAPI** (`app/main.py`)  
   * `POST /predict` – JSON ▶ решение + вероятность  
   * Валидация входа через Pydantic, встроенная предобработка.  
7. **Деплой** – EC2 скрипт / Terraform / Elastic Beanstalk.

---

POST /predict
Content-Type: application/json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 750000.0,
  "loan_amount": 250000.0,
  "loan_term": 36,
  "cibil_score": 760,
  "residential_assets_value": 1000000.0,
  "commercial_assets_value": 0.0,
  "luxury_assets_value": 150000.0,
  "bank_asset_value": 500000.0
}

{
  "prediction": "Approved",
  "probability": 0.97
}


📊 Результаты
Модель	Accuracy	Precision	Recall	F1	ROC AUC
Логистическая регрессия	0.93	0.92	0.94	0.93	0.96
Random Forest ★	0.97	0.96	0.98	0.97	0.99
SVM	0.95	0.94	0.95	0.94	0.97
Decision Tree	0.91	0.90	0.92	0.91	0.92
KNN	0.88	0.87	0.88	0.87	0.90

Победила Random Forest (тюнинг параметров).
Топ-3 важных признака: cibil_score, income_annum, loan_amount.

🛠️ Технологии
Python 3.10, pandas, scikit-learn, matplotlib, seaborn

FastAPI, uvicorn, pydantic

joblib для сериализации

Автор
Алымбек Ибрагимов – Python-разработчик, начинающий ML Engineer
LinkedIn: (https://www.linkedin.com/in/alymbek-ibragimov-447876336/) • GitHub: (https://github.com/Alymbaek)

