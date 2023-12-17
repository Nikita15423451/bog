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

@dp.message_handler(commands=['start'])
async def start(message: aiogram.types.Message):
    keyboard = aiogram.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(aiogram.types.KeyboardButton('Правда'))
    keyboard.add(aiogram.types.KeyboardButton('Ложь'))
    await message.answer("Привет! Нажми на кнопку, чтобы начать обучение модели.", reply_markup=keyboard)

@dp.message_handler(lambda message: message.text in ['Правда', 'Ложь'])
async def handle_feedback(message: aiogram.types.Message):
    feedback = message.text.lower()
    conversations.append(feedback)

    # Преобразование данных для обучения модели
    labels = np.array([1 if x == 'правда' else 0 for x in conversations])

    # Инициализация токенизатора и преобразование текста в числовой формат
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(conversations)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = tokenizer.texts_to_sequences(conversations)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X_train = input_sequences

    # Создание и обучение модели на основе данных
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, labels, epochs=10, batch_size=32)

    await message.answer(f"Вы выбрали: {feedback}. Модель обучена на вашем ответе.")

if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
