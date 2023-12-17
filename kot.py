import aiogram
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Инициализация бота
bot = aiogram.Bot(token="YOUR_BOT_TOKEN_HERE")
dp = aiogram.Dispatcher(bot)

# Ваши фразы для обучения модели
conversations = [
    "Привет"
]

# Дополнительные фразы для обучения модели
additional_conversations = [
    # Дополнительные разговоры для обогащения данных
]

# Объединение начальных и новых данных для обучения
conversations.extend(additional_conversations)

# Преобразование текста в последовательности чисел
tokenizer = Tokenizer()
tokenizer.fit_on_texts(conversations)
total_words = len(tokenizer.word_index) + 1

max_sequence_len = max([len(x.split()) for x in conversations])
input_sequences = []
for line in conversations:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Проверка данных перед обучением
if input_sequences.size == 0:
    print("Данные для обучения пустые. Пожалуйста, добавьте данные для обучения.")
    exit()

# Формирование xs и ys из input_sequences
xs = input_sequences[:, :-1]
ys = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)

# Проверка размерности данных перед обучением
if xs.size == 0 or ys.size == 0:
    print("Формат данных для обучения некорректен. Пожалуйста, проверьте данные перед обучением.")
    exit()

# Определение модели
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    # Добавьте больше слоев LSTM, Dropout и другие слои по вашему выбору
    # Пример:
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xs, ys, epochs=5000, verbose=1, batch_size=64)

# Остальной код для бота
# ...



async def generate_response(seed_text):
    # Ваш код для генерации ответа на основе обученной модели
    # seed_text - текст, с которого начинается генерация
    # Верните сгенерированный текст как ответ
    return "Результат генерации"

@dp.message_handler()
async def echo(message: aiogram.types.Message):
    user_input = message.text  # Получаем текст от пользователя
    if user_input.lower() == 'exit':
        await message.reply("Пока!")
        return

    generated_response = await generate_response(user_input)  # Генерируем ответ на основе введенного текста
    await message.reply(generated_response)  # Отправляем сгенерированный ответ пользователю

if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
