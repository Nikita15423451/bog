import aiogram
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Подготовка данных для обучения seq2seq модели
conversations = [
    ("Привет!", "Привет, как дела?"),
    ("Неплохо, а у тебя?", "У меня все хорошо."),
    ("Как тебя зовут?", "Адам."), 
    ("Кто ты?", "Искуственный интелект"), 
    ("Что ты можешь?", "Много чего"), 
]

questions = [x[0] for x in conversations]
answers = [x[1] for x in conversations]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

tokenizer.word_index['<start>'] = total_words
tokenizer.word_index['<end>'] = total_words + 1
total_words += 2

# Добавим эквивалент для индекса 57
tokenizer.index_word[57] = 'новое_слово'

# Изменяем индексы, добавив два новых слова
tokenizer.word_index['новое_слово'] = 57
total_words += 1

tokenized_questions = tokenizer.texts_to_sequences(questions)
tokenized_answers = tokenizer.texts_to_sequences(answers)

max_sequence_len = max([len(x) for x in tokenized_questions + tokenized_answers])
encoder_input_data = pad_sequences(tokenized_questions, maxlen=max_sequence_len, padding='post')
decoder_input_data = pad_sequences(tokenized_answers, maxlen=max_sequence_len, padding='post')
decoder_output_data = np.zeros_like(decoder_input_data)

for i, seq in enumerate(tokenized_answers):
    decoder_output_data[i, 0:len(seq)] = seq

# Создание модели seq2seq
embedding_dim = 64
hidden_units = 128

# Остальная часть кода остаётся без изменений
# ...

# Обновление токенизатора после изменения словаря
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

tokenizer.word_index['<start>'] = total_words
tokenizer.word_index['<end>'] = total_words + 1
total_words += 2

# Пересоздание модели seq2seq и последующее обучение
# ...

# Инициализация бота и обработка сообщений
# ...

# Инициализация бота и обработка сообщений
bot = aiogram.Bot(token="6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0")
dp = aiogram.Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def start(message: aiogram.types.Message):
    await message.answer("Привет! Чтобы начать общение, отправьте свое сообщение.")

@dp.message_handler()
async def handle_messages(message: aiogram.types.Message):
    input_text = message.text.lower()
    response = await generate_response(input_text)
    await message.answer(response)

# Запуск бота
if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
