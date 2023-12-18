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

for i in range(100):
    encoder_texts.append(f"Пример входного предложения {i}")
    decoder_texts.append(f"Пример целевого предложения {i}")

tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(encoder_texts)
num_encoder_tokens = len(tokenizer_encoder.word_index) + 1
max_encoder_seq_length = max([len(text.split()) for text in encoder_texts])

tokenizer_decoder = Tokenizer()
tokenizer_decoder.fit_on_texts(decoder_texts)
if '\t' not in tokenizer_decoder.word_index:
    tokenizer_decoder.word_index['\t'] = len(tokenizer_decoder.word_index) + 1
num_decoder_tokens = len(tokenizer_decoder.word_index) + 1
max_decoder_seq_length = max([len(text.split()) for text in decoder_texts])

encoder_sequences = tokenizer_encoder.texts_to_sequences(encoder_texts)
decoder_sequences = tokenizer_decoder.texts_to_sequences(decoder_texts)

encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=max_decoder_seq_length, padding='post')

decoder_output_data = np.zeros((len(decoder_sequences), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, seq in enumerate(decoder_sequences):
    for j, token in enumerate(seq):
        if j > 0:
            decoder_output_data[i][j - 1][token] = 1.0

latent_dim = 256

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

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_pred = decoder_embedding

decoder_outputs_pred, state_h_pred, state_c_pred = decoder_lstm(
    decoder_embedding_pred, initial_state=decoder_states_inputs)
decoder_states_pred = [state_h_pred, state_c_pred]
decoder_outputs_pred = decoder_dense(decoder_outputs_pred)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_pred] + decoder_states_pred
)

async def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_decoder.word_index['\t']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in tokenizer_decoder.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word is not None:
            if sampled_word == '\n':
                stop_condition = True
            else:
                decoded_sentence += sampled_word + ' '

        if len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence


bot = Bot(token='6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0')
dp = Dispatcher(bot)

@dp.message_handler(content_types=ContentType.TEXT)
async def process_text_messages(message: types.Message):
    input_seq = pad_sequences(tokenizer_encoder.texts_to_sequences([message.text]), maxlen=max_encoder_seq_length, padding='post')
    decoded_sentence = await decode_sequence(input_seq)
    response_text = f"Ваш запрос: {message.text}\nОтвет: {decoded_sentence}"

    await message.answer(response_text)

if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)
