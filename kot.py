import aiogram
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Инициализация бота
bot = aiogram.Bot(token="6330276709:AAGJWL3LpSrKlOG-Xya6lhPUG-p9uN-C9nQ")
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
    "Некит", 
    "Привет", 
    "Занят?", 
    "Что делаешь?", 
    "Здорова", 
    "Ты выйдешь гулять?", 
    "Сейчас?", 
    "Да", 
    "Могу", 
    "Вечером пиздец холодно", 
    "Ага", 
    "Окей", 
    "Ты где?", 
    "Пока на хате", 
    "Ок", 
    "Так че завтра идём на работу?", 
    "Да", 
    "Бля завтра дождь", 
    "Прямо с самого утра?", 
    "С 9", 
    "Доброе утро", 
    "Спишь?", 
    "Нет", 
    "Выходи", 
    "Я возле тебя"
    "Щас выйду"
    "А", 
    "Я сейчас поем", 
    "Ок", 
    "Я про коноплянную настойку", 
    "Продать можно намного больше людей", 
    "Скажи то что отмазаться перед ней, поэтому сказал типо приехали", 
    "Хорошо", 
    "Та помолчи на счёт 0.5", 
    "Мама твоя слышала", 
    "Да похуй", 
    "0.5 можно", 
    "И что ты скажешь, типо бухим ездил до неё?", 
    "Рд", 
    "Нет трезвый ,а потом выпел", 
    "Родители", 
    "Пон", 
    "Ну что выпьем?", 
    "Перед братом?", 
    "Что ты доебался до него", 
    "Чтоб не раслоблялся", 
    "Налевай", 
    "Некит", 
    "Ты где?", 
    "В шараге", 
    "Я ещё на хате", 
    "Понятно", 
    "Привет", 
    "Занят?", 
    "Уже нет", 
    "Нет", 
    "А что?", 
    "А где?", 
    "Я на стадике", 
    "Че там народа много?", 
    "Да не особо", 
    "Понятно", 
    "Шабашку нужно искать" 
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
xs, ys = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)

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

async def generate_response(seed_text):
    generated_response = seed_text
    while True:
        input_sequence = tokenizer.texts_to_sequences([generated_response])[0]
        input_sequence = np.array(pad_sequences([input_sequence], maxlen=max_sequence_len-1, padding='pre'))
        predicted_output = model.predict(input_sequence)
        
        predicted_word_index = np.argmax(predicted_output, axis=-1)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]

        generated_response += " " + predicted_word
        
        if predicted_word == '.' or len(generated_response.split()) >= 30:  # Условие останова
            break

    return generated_response

@dp.message_handler()
async def echo(message: aiogram.types.Message):
    user_input = message.text
    if user_input.lower() == 'exit':
        await message.reply("Пока!")
        return

    generated_response = await generate_response(user_input)
    await message.reply(generated_response)

if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
