import aiogram
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Инициализация бота
bot = aiogram.Bot(token="6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0")
dp = aiogram.Dispatcher(bot)

# Ваши фразы для обучения модели
conversations = [
        "Привет",
    "Как дела?",
    "Что нового?",
    "Здравствуйте",
    "Приветствую!",
    "Доброе утро", 
    "Ты на хате?", 
    "Ты дома?", 
    "Да", 
    "Заебись", 
    "У тебя родичи дома?", 
    "Мама", 
    "Пон, я хочу у Дена спросить чтоб габы штук 20 дал", 
    "Давай", 
    "Бля он только 10 может дать больше у нкго нет", 
    "У нас там осталось ещё?", 
    "Некит", 
    "Привет", 
    "Спишь", 
]

# Дополнительные фразы для обучения модели
additional_conversations = [
    "?", 
    "Короче есть предложение у тебя купят карту но в Сбербанк не будут заходить им она нужна просто снимать бабки и всё, он сказал что 10 к только готов дать", 
    "Короче он сам тебе там объяснит что да как", 
    "Когда ему понадобится", 
    "Там торгуется сам с ним", 
    "15", 
    "К", 
    "Я готов продать", 
    "Ему не седня надо", 
    "Он мне скажет когда", 
    "Хорошо", 
    "Выходить будешь?", 
    "Да", 
    "Окей, сейчас я оденусь и выйду", 
    "Нужно шабашку найти", 
    "Надо", 
    "А что на счëт карты", 
    "Он ещё ничего не сказал", 
    "Ну я хз ему пока не нужна", 
    "На счëт работы я разберусь", 
    "Че ты где?", 
    "На пятом", 
    "Ясно", 
    "Некит", 
    "Спишь?", 
    "Выходи", 
    "Ты где?", 
    "Возле твоего дома", 
    "Сможешь меня за сигами отвезти?", 
    "Че ты ещё в красе?", 
    "Ау", 
    "Некит",     # Дополнительные разговоры для обогащения данных

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

# Формирование xs и ys из input_sequences
xs = input_sequences[:, :-1]
ys = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(xs.shape, ys.shape)  # Проверка размера данных перед обучением

model.fit(xs, ys, epochs=500, verbose=1, batch_size=64)

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
