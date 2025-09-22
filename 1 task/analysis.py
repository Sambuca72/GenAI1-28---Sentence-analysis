import matplotlib # Библиотека для построения графиков
matplotlib.use("Agg") # Используем безоконный бэкенд (график сохраняется в файл, без показа окна)

import matplotlib.pyplot as plt # matplotlib - построение графиков через pyplot
import numpy as np # numpy - работа с массивами и вычисление среднего

import nltk # nltk - инструменты для обработки естественного языка
from nltk.tokenize import sent_tokenize, word_tokenize # Готовые функции для разбиения текста на предложения и слова

from collections import Counter # Counter - удобный счётчик для подсчёта одинаковых значений
import os # os - работа с файловой системой (например, проверка наличия файлов)
nltk.download("punkt", quiet=True) # Загружаем модель "punkt" для разбиения текста

LANG = "russian" # Язык токенизации
INPUT_FILE = "input.txt" # Имя файла с текстом, если есть
OUTPUT_FILE = "hist.png" # Имя файла для сохранения графика

# 0) Задаём текст для обработки длины
os.path.exists(INPUT_FILE)
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# 1) Разбиваем исходный текст на отдельные предложения т.е список строк
sentences = sent_tokenize(text, language=LANG)

# 2) Считаем длину каждого предложения
# Здесь мы используем word_tokenize, чтобы разбить предложение на токены
# По умолчанию туда попадают и слова, и знаки препинания, и числа
# Чтобы считать только слова надо:
# 1.t.isalpha() оставляет только токены, состоящие из букв
# 2.Пунктуация и числа будут отсеяны
lengths = []
for i, s in enumerate(sentences, 1):
    words_only = [t for t in word_tokenize(s, language=LANG) if t.isalpha()]
    length = len(words_only)
    lengths.append(length)
    print(f"{i}) {length} слов — {s}")

# 3) Гистограмма: готовим данные для построения графика распределения
if lengths: # Если есть хотя бы одно непустое предложение
    counter = Counter(lengths) # Считаем, сколько предложений оказалось длиной 4 слова, 5 слов, 6 слов и т.д.
    
    # Задаем параметры для гистограммы
    plt.bar(counter.keys(), counter.values(), width=0.05, edgecolor="black")
    plt.title("Распределение длины предложений")
    plt.xlabel("Количество слов в предложении")
    plt.ylabel("Частота (количество одинаковых предложений по длине)")
    plt.xticks(range(min(lengths), max(lengths) + 1))
else:
    # если текст пуст - делаем пустой график
    plt.figure()
    plt.title("Распределение длины предложений (нет данных)")
    print("Нет текста")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close()

# 4) Расчёт средней длины предложения
avg_len = float(np.mean(lengths)) if lengths else 0.0
print("Средняя длина предложения:", round(avg_len, 2))
