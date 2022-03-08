import json
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import strings as st


def train():
    config_file = open(st.BOT_CONFIG_FILENAME, "r")  # открытие файла
    BOT_CONFIG = json.load(config_file)  # преобразование из json в структуру данных
    config_file.close()

    # тексты
    x = []
    # классы
    y = []
    # задача модели - по х находить y
    for name, data in BOT_CONFIG["intents"].items():
        for example in data["examples"]:
            x.append(example)  # собираем тексты в Х
            y.append(name)  # собираем классы в y

    # x[-5:]#вывод 5 последних

    vectorizer = CountVectorizer()  # можно указать настройки
    vectorizer.fit(x)  # передаем набор текстов, чтобы векторайзер их проанализтровал

    #print(vectorizer.vocabulary_)

    x_vectorized = vectorizer.transform(x)  # трансформируем тексты в вектора (наборы чисел)

    v = open(st.BOT_VECTORIZER_FILENAME, "wb")
    pickle.dump(vectorizer, v)
    v.close()

    model = LogisticRegression()  # Настройки
    model.fit(x_vectorized, y)  # Модель научиться по Х определять

    f = open(st.BOT_MODEL_FILENAME, "wb")
    pickle.dump(model, f)
    f.close()


