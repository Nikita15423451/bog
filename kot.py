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
