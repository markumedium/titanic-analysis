# 🚢 Titanic Survival Analysis

Учебный проект по машинному обучению на данных пассажиров Титаника.  
Цель — предсказать выживаемость пассажиров на основе их характеристик.

![titanic](https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg)

---

## 📊 Используемые технологии

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression)
- Git, GitHub

---

## 📁 Структура проекта

```
TitanicAnalysis/
├── data/               # Данные (исключены из Git)
├── titanic_analysis.py # Основной скрипт анализа
├── requirements.txt    # Зависимости
├── .gitignore          # Исключения
└── README.md           # Описание проекта
```

---

## 🚀 Как запустить

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/markumedium/titanic-analysis.git
   cd titanic-analysis
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Скачайте `train.csv` с [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)  
   и поместите его в папку `data/`

4. Запустите анализ:
   ```bash
   python titanic_analysis.py
   ```

---

## ✅ Результаты

- Модель: Logistic Regression
- Точность: ~78%
- Основные выводы:
  - Женщины выживали чаще мужчин
  - Пассажиры 1-го класса имели лучшие шансы
  - Дети выживали чаще взрослых

---

## 🧠 Возможные улучшения

- Добавить больше признаков: `SibSp`, `Parch`
- Попробовать другие модели: RandomForest, XGBoost
- Сделать Streamlit-приложение
- Настроить кросс-валидацию и подбор гиперпараметров

---

## 👤 Автор

[markumedium](https://github.com/markumedium)
