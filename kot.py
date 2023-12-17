import aiogram
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Инициализация бота
bot = aiogram.Bot(token="6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0")
dp = aiogram.Dispatcher(bot)

# Данные для обучения модели
conversations = []

def train_model():
    global conversations
    
    # Здесь загружаются и предобрабатываются ваши реальные данные
    # Предположим, что у вас есть файл с текстовыми сообщениями

    # Пример чтения данных из файла (замените этот блок кода на ваш реальный процесс загрузки данных)
    with open('bat.txt', 'r', encoding='utf-8') as file:
        conversations = file.readlines()
    
    # Ваш код предобработки текстовых данных
    # Например, токенизация, очистка текста от шума, приведение к нижнему регистру и т.д.

    labels = np.array([1 if x == 'Правда\n' else 0 for x in conversations])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(conversations)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = tokenizer.texts_to_sequences(conversations)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(input_sequences, labels, epochs=10, batch_size=32)
    return model

# Обучение модели при запуске скрипта
trained_model = train_model()

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def start(message: aiogram.types.Message):
    keyboard = aiogram.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(aiogram.types.KeyboardButton('Правда'))
    keyboard.add(aiogram.types.KeyboardButton('Ложь'))
    await message.answer("Привет! Нажми на кнопку, чтобы начать обучение модели.", reply_markup=keyboard)

# Обработчик ответа пользователя "Правда" или "Ложь"
@dp.message_handler(lambda message: message.text in ['Правда', 'Ложь'])
async def handle_feedback(message: aiogram.types.Message):
    feedback = message.text.lower()
    conversations.append(feedback)

    # Обучение модели при получении новых данных
    global trained_model
    trained_model = train_model()  # Переобучение модели на новых данных

    await message.answer(f"Вы выбрали: {feedback}. Модель обучена на вашем ответе.")

# Обработчик всех остальных сообщений от пользователя
@dp.message_handler()
async def handle_other_messages(message: aiogram.types.Message):
    global trained_model
    if trained_model is not None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(conversations)
        max_sequence_len = max([len(seq.split()) for seq in conversations])

        user_input = message.text.lower()
        encoded_input = tokenizer.texts_to_sequences([user_input])
        padded_input = pad_sequences(encoded_input, maxlen=max_sequence_len, padding='pre')

        prediction = trained_model.predict(padded_input)
        # Ваша логика обработки предсказания и генерации ответа
        # Например:
        if prediction > 0.5:
            response = "Правда"
        else:
            response = "Ложь"

        await message.answer(f"Модель предсказывает: {response}")

if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
