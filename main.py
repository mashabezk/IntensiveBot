import json  # импорт из библиотеки Json
import os
import pickle
import random
import re

import nltk
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

import strings as st
import modelTraining as mt

# BotFather
f = open(st.TOKEN_FILENAME, "r")
BOT_KEY = f.read()

config_file = open(st.BOT_CONFIG_FILENAME, "r")  # открытие файла
BOT_CONFIG = json.load(config_file)  # преобразование из json в структуру данных
config_file.close()


# создаем функцию, которая выкинет знаки препинания и приведет текст к нижнему регистнру
def filter(text):
    text = text.lower()  # текст в нижний регистр
    punctuation = r"[^\w\s]"  # удалить все знаки препинания; ^- все кроме, \w-буквы, \s-пробелы
    return re.sub(punctuation, "", text)  # заменяем знаки препинания на пустоту


# обьявление функции, которая посчитает похожи ли тексты
def isMatching(text1, text2):
    text1 = filter(text1)
    text2 = filter(text2)
    distance = nltk.edit_distance(text1, text2)  # посчитаем расстояние между текстами
    average_length = (len(text1) + len(text2)) / 2  # Посчитаем среднюю длину текстов
    return distance / average_length  # примерно насколько тексты отличаются в процентах


def getIntent(text):  # понимать намерение по тексту
    all_intents = BOT_CONFIG["intents"]
    for name, data in all_intents.items():  # пройти по всем намерениям в name , и остальное в переменную data
        for example in data["examples"]:  # пройти по всем намерениям intent, и положить текст вперемнную example
            if isMatching(text, example) < 0.4:  # если текст совпадаетс примером
                return name


def getAnswer(intent):  # получить ответ
    responses = BOT_CONFIG["intents"][intent]["responses"]
    return random.choice(responses)


def bot(text):  # функция=бот
    intent = getIntent(text)

    if not intent:  # если намерение не найдено
        test = vectorizer.transform([text])
        intent = model.predict(test)[
            0]  # По Х предсказать у, т.е. классифицировать #подключить модель машинного обучения(классификатор текстов)

    #print("Intent = ", intent)

    if intent:  # если намерение найдено - выдать ответ
        return getAnswer(intent)

    # Заглушка
    failure_phrases = BOT_CONFIG["failure_phrases"]
    return random.choice(failure_phrases)


if not os.path.exists(st.BOT_VECTORIZER_FILENAME) or not os.path.exists(st.BOT_MODEL_FILENAME):
    print("запускается модель обучения")
    mt.train()  # запустить модель обучения
    print("обучение завершено")

v = open(st.BOT_VECTORIZER_FILENAME, "rb")
vectorizer = pickle.load(v)
v.close()

f = open(st.BOT_MODEL_FILENAME, "rb")
model = pickle.load(f)
f.close()


def hello(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Hello {update.effective_user.first_name}')


# функция будет вызвана при получении сообщения
def botMessage(update: Update, context: CallbackContext):
    text = update.message.text  # что нам написал пользователь
    reply = bot(text)  # готовим ответ
    update.message.reply_text(reply)  # отправляем ответ пользователю


updater = Updater(BOT_KEY)
print("бот запущен")

updater.dispatcher.add_handler(
    CommandHandler('hello', hello))  # конфигурация, при подключении команды hello вызвать функцию hello
updater.dispatcher.add_handler(MessageHandler(Filters.text,
                                              botMessage))  # конфигурация, при получении текстового сообщения будет вызвана функция botMessage

updater.start_polling()
updater.idle()
