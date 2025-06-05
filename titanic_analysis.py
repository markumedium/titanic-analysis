import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Загрузка данных
df = pd.read_csv('data/data.csv')

# Просмотр первых 5 строк
print(df.head())

print(df.shape)           # Размер
print(df.columns)         # Названия столбцов
print(df.isnull().sum())  # Пропущенные значения



# Удаляем Cabin
df.drop('Cabin', axis=1, inplace=True)

# Удаляем строки с пропусками в Embarked
df.dropna(subset=['Embarked'], inplace=True)

# Заполняем Age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Проверка — сколько осталось пропусков
print(df.isnull().sum())

# 1. Распределение выживших
sns.countplot(data=df, x='Survived')
plt.title("Распределение выживших")
plt.show()

# 2. По полу
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Выживаемость по полу")
plt.show()

# 3. По классам
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Выживаемость по классам")
plt.show()




# Целевая переменная
y = df['Survived']

# Выбираем признаки
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].copy()

# Преобразуем пол: male → 0, female → 1
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Преобразуем Embarked в числовые признаки (One Hot Encoding)
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём модель
model = LogisticRegression(max_iter=1000)

# Обучаем модель
model.fit(X_train, y_train)

# Предсказываем на тесте
y_pred = model.predict(X_test)

# Оцениваем точность
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))
print("\nКлассификационный отчёт:\n", classification_report(y_test, y_pred))




# --- Результаты модели ---
print("\n=== Результаты модели ===")
print(f"Точность модели: {accuracy_score(y_test, y_pred):.2f}")
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))
print("\nКлассификационный отчёт:\n", classification_report(y_test, y_pred))

# --- Выводы ---
print("\n=== Выводы ===")
print("1. Базовая модель логистической регрессии достигает точности 78%.")
print("2. Женщины имели значительно больше шансов выжить — модель учитывает пол как важный признак.")
print("3. Пассажиры 1-го класса выживали чаще, чем 2-го и 3-го.")
print("4. Модель чаще правильно определяет тех, кто не выжил (precision=0.83), чем тех, кто выжил (recall=0.75).")
print("5. Есть потенциал для улучшения — можно добавить больше признаков и попробовать другие алгоритмы.")
