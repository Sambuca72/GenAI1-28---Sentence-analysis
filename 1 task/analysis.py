import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # matplotlib — построение графиков 
import numpy as np  # numpy — работа с массивами и вычисление среднего

# nltk — обработка естественного языка
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("punkt", quiet=True)  # Загружаем модель "punkt" для разбиения текста на предложения/слова

# 0) Задаём текст для обработки длины
text = ("Искусственный интеллект активно развивается. "
        "Он используется для создания текста, изображений и музыки. "
        "Такие модели, как ChatGPT, уже стали частью повседневной жизни. "
        "Студенты анализируют длины предложений. "
        "А ещё они строят гистограмму и считают среднюю длину. ")

# 1) Разбиваем исходный текст на отдельные предложения т.е список строк
sentences = sent_tokenize(text, language="russian")

# 2) Считаем длину каждого предложения
# Здесь мы используем word_tokenize, чтобы разбить предложение на токены
# По умолчанию туда попадают и слова, и знаки препинания, и числа
# Чтобы считать только слова надо:
# 1.t.isalpha() оставляет только токены, состоящие из букв
# 2.Пунктуация и числа будут отсеяны
lengths = []
for i, s in enumerate(sentences, 1):
    words_only = [t for t in word_tokenize(s, language="russian") if t.isalpha()]
    length = len(words_only)
    lengths.append(length)
    print(f"{i}) {length} слов — {s}")

# 3) Гистограмма: готовим данные для построения графика распределения
from collections import Counter

if lengths:  # Если есть хотя бы одно непустое предложение
    counter = Counter(lengths) # Считаем, сколько предложений оказалось длиной 4 слова, 5 слов, 6 слов и т.д.

    # Задаем параметры для гистограммы
    plt.bar(counter.keys(), counter.values(), width=0.05, edgecolor="black")
    plt.title("Распределение длины предложений")
    plt.xlabel("Количество слов в предложении")
    plt.ylabel("Частота (количество одинаковых предложений по длине)")
    plt.xticks(range(min(lengths), max(lengths) + 1))
else:
    # если текст пуст или все предложения пустые → делаем пустой график
    plt.figure()
    plt.title("Распределение длины предложений (нет данных)")
    print("Нет текста")

plt.tight_layout()
plt.savefig("hist.png", dpi=150, bbox_inches="tight")
plt.close()

# 4) Расчёт средней длины предложения
avg_len = float(np.mean(lengths)) if lengths else 0.0
print("Средняя длина предложения:", round(avg_len, 2))

