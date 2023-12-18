import aiogram
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import ContentType
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Пример данных, включая 100 предложений
encoder_texts = [
    'Как дела?',
    'Привет, что нового?',
    'Чем занимаешься?',
    # Дополнительные предложения...
]

decoder_texts = [
    'Прекрасно!',
    'Ничего особенного.',
    'Читаю книгу.',
    # Дополнительные предложения...
]

# Генерация большого количества предложений для обогащения словаря
for i in range(100):
    encoder_texts.append(f"Example input sentence {i}")
    decoder_texts.append(f"Example target sentence {i}")

# Определение параметров модели
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(encoder_texts)
num_encoder_tokens = len(tokenizer_encoder.word_index) + 1
max_encoder_seq_length = max([len(text.split()) for text in encoder_texts])

tokenizer_decoder = Tokenizer()
tokenizer_decoder.fit_on_texts(decoder_texts)
num_decoder_tokens = len(tokenizer_decoder.word_index) + 1
max_decoder_seq_length = max([len(text.split()) for text in decoder_texts])

# Преобразование текстов в последовательности чисел (токены)
encoder_sequences = tokenizer_encoder.texts_to_sequences(encoder_texts)
decoder_sequences = tokenizer_decoder.texts_to_sequences(decoder_texts)

# Добавление паддинга для равной длины последовательностей
encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=max_decoder_seq_length, padding='post')

# Создание decoder_output_data (для обучения модели)
decoder_output_data = np.zeros((len(decoder_sequences), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, seq in enumerate(decoder_sequences):
    for j, token in enumerate(seq):
        if j > 0:
            decoder_output_data[i][j - 1][token] = 1.0

# Создание модели seq2seq
latent_dim = 256  # Примерный размер скрытого слоя

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=64, epochs=50, validation_split=0.2)

# Определение функции decode_sequence
async def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_decoder.word_index['\t']  # Замена '\t' токена начала последовательности

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Выбор индекса токена с наибольшей вероятностью
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in tokenizer_decoder.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word == '\n':  # Проверка на символ конца последовательности
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '

        if len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True

        # Обновление последовательности для предсказания следующего токена
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Обновление состояний LSTM
        states_value = [h, c]

    return decoded_sentence

# Функция для обработки входящих текстовых сообщений
async def generate_response(message: types.Message):
    input_text = message.text
    input_seq = tokenizer_encoder.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
    decoded_sentence = await decode_sequence(input_seq)

    response_text = f"Input: {input_text}\nResponse: {decoded_sentence}"

    # Отправка ответа обратно пользователю
    await message.answer(response_text)

# Создание экземпляра бота и диспетчера
bot = Bot(token='6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0')
dp = Dispatcher(bot)

# Функция для обработки входящих сообщений
@dp.message_handler(content_types=ContentType.TEXT)
async def process_text_messages(message: types.Message):
    await generate_response(message)

# Запус
# Запуск бота
if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)
