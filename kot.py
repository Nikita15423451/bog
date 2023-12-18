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
    1: 'Как твои дела?',
    2: 'Что нового?',
    3: 'Чем занимаешься?',
    4: 'Как настроение?',
    5: 'Что ты думаешь о...?',
    6: 'Что ты любишь делать в свободное время?',
    7: 'Как прошел твой день?',
    8: 'Что тебя вдохновляет?',
    9: 'Что тебе нравится больше всего в учебе/работе?',
    10: 'Что бы ты хотел(а) изменить в своей жизни?',
    11: 'Как проводишь выходные?',
    12: 'Что тебе нравится готовить?',
    13: 'Какие у тебя увлечения?',
    14: 'Какие книги ты читаешь в последнее время?',
    15: 'Что тебя сейчас волнует?',
    16: 'Чем занимаешься в свободное время?',
    17: 'Как ты отдыхаешь после работы/учебы?',
    18: 'Как ты улучшаешь свои навыки?',
    19: 'Что ты любишь делать на выходных?',
    20: 'Какие у тебя планы на будущее?',
    21: 'Что ты думаешь об экологии?',
    22: 'Как ты относишься к спорту?',
    23: 'Что было самым важным для тебя в этом году?',
    24: 'Что тебе интересно изучать в данный момент?',
    25: 'Как ты развиваешься как личность?',
    26: 'Какой был твой самый удачный день?',
    27: 'Что тебе помогает быть в хорошем настроении?',
    28: 'Как ты относишься к путешествиям?',
    29: 'Какие у тебя любимые фильмы/сериалы?',
    30: 'Что ты делаешь для сохранения здоровья?',
    31: 'Как ты относишься к новым технологиям?',
    32: 'Какие книги ты рекомендуешь для чтения?',
    33: 'Что бы ты сделал(а), если бы получил(а) деньги в подарок?',
    34: 'Как ты планируешь свой день?',
    35: 'Что тебя вдохновляет на саморазвитие?',
    36: 'Какой был твой самый важный урок в жизни?',
    37: 'Что ты обычно делаешь перед сном?',
    38: 'Что ты любишь делать на природе?',
    39: 'Что ты думаешь о дружбе?',
    40: 'Что бы ты сделал(а) на месте президента?',
    41: 'Что ты делаешь для поддержания баланса между работой и личной жизнью?',
    42: 'Что бы ты хотел(а) изучить новое?',
    43: 'Что ты ценишь в друзьях?',
    44: 'Что ты думаешь о саморазвитии?',
    45: 'Какие у тебя цели на ближайший год?',
    46: 'Как ты относишься к искусству?',
    47: 'Что ты считаешь главным в жизни?',
    48: 'Что ты любишь в себе больше всего?'
    # Дополнительные предложения...
]

decoder_texts = [
    1: 'Хорошо, спасибо!',
    2: 'Ничего особенного.',
    3: 'Читаю книгу.',
    4: 'Отлично!',
    5: 'Это сложный вопрос.',
    6: 'Люблю заниматься спортом.',
    7: 'День был насыщенным.',
    8: 'Музыка меня вдохновляет.',
    9: 'Мне нравится учить новые вещи.',
    10: 'Хочу изменить свои привычки.',
    11: 'Обычно провожу время с друзьями.',
    12: 'Готовлю разные блюда.',
    13: 'Увлекаюсь фотографией.',
    14: 'Читаю фантастику.',
    15: 'Думаю о своих планах на будущее.',
    16: 'Играю на музыкальных инструментах.',
    17: 'Смотрю фильмы или занимаюсь хобби.',
    18: 'Посещаю курсы и тренинги.',
    19: 'Путешествую.',
    20: 'Планирую развитие карьеры.',
    21: 'Я за экологию.',
    22: 'Спорт важен для здоровья.',
    23: 'Личные достижения.',
    24: 'Изучаю новые языки.',
    25: 'Стремлюсь к самосовершенствованию.',
    26: 'Это был день успехов.',
    27: 'Общение с близкими помогает.',
    28: 'Обожаю путешествия.',
    29: 'Мои любимые фильмы - это...',
    30: 'Занимаюсь спортом и правильным питанием.',
    31: 'Я в теме новинок.',
    32: 'Рекомендую классику.',
    33: 'Инвестирую во что-то полезное.',
    34: 'Планирую дела на день.',
    35: 'Успехи других людей.',
    36: 'Важно учиться на ошибках.',
    37: 'Читаю книгу или слушаю музыку.',
    38: 'Гуляю на свежем воздухе.',
    39: 'Дружба - это важно.',
    40: 'Решал бы проблемы страны.',
    41: 'Стараюсь делить время на работу и личные увлечения.',
    42: 'Хочу изучить новый язык.',
    43: 'Честность и преданность.',
    44: 'Считаю, что это важно для личностного роста.',
    45: 'Улучшение здоровья и карьеры.',
    46: 'Искусство вдохновляет.',
    47: 'Семья и близкие отношения.',
    48: 'Люблю свою настойчивость.'
    # Дополнительные предложения...
]
    # Дополнительные предложения...


for i in range(48):
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
        sampled_word = tokenizer_decoder.index_word.get(sampled_token_index, None)

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
