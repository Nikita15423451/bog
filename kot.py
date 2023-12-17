import aiogram
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ... (Код для создания модели seq2seq, обработки данных и функции generate_response)


# Подготовка данных
# Примеры диалогов для обучения модели
conversations = [
    ("Привет!", "Привет, как дела?"),
    ("Неплохо, а у тебя?", "У меня все хорошо."),
    # Другие диалоги...
]

# Создание набора вопросов и ответов
questions = [x[0] for x in conversations]
answers = [x[1] for x in conversations]

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

# Преобразование текста в последовательности чисел
tokenized_questions = tokenizer.texts_to_sequences(questions)
tokenized_answers = tokenizer.texts_to_sequences(answers)

# Подготовка входных и выходных данных
max_sequence_len = max([len(x) for x in tokenized_questions + tokenized_answers])
encoder_input_data = pad_sequences(tokenized_questions, maxlen=max_sequence_len, padding='post')
decoder_input_data = pad_sequences(tokenized_answers, maxlen=max_sequence_len, padding='post')

# Создание модели seq2seq с механизмом внимания
embedding_dim = 64
hidden_units = 128

encoder_inputs = tf.keras.layers.Input(shape=(max_sequence_len,))
encoder_embedding = tf.keras.layers.Embedding(total_words, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(max_sequence_len,))
decoder_embedding = tf.keras.layers.Embedding(total_words, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

attention = tf.keras.layers.Attention()
attention_output = attention([decoder_outputs, encoder_outputs])

decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention_output])
decoder_dense = tf.keras.layers.Dense(total_words, activation='softmax')
output = decoder_dense(decoder_concat_input)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit([encoder_input_data, decoder_input_data], np.expand_dims(tokenized_answers, -1), batch_size=64, epochs=10)

# Использование модели для генерации ответа
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_len, padding='post')
    
    states_value = encoder_lstm.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_lstm.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]
        
        if sampled_word != '<end>':
            decoded_sentence += sampled_word + ' '
            
        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_sequence_len:
            stop_condition = True
            
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        states_value = [h, c]
    
    return decoded_sentence

# Пример использования модели для генерации ответа
input_text = "Привет!"
response = generate_response(input_text)
print(f"Input: {input_text}\nGenerated Response: {response}")

# Инициализация бота и диспетчера
bot = aiogram.Bot(token="6439522576:AAGBJahBMqhUDlaikziF3Dqm3lEdE4a6mL0")
dp = aiogram.Dispatcher(bot)

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def start(message: aiogram.types.Message):
    await message.answer("Привет! Чтобы начать общение, отправьте свое сообщение.")

# Обработчик всех остальных сообщений от пользователя
@dp.message_handler()
async def handle_messages(message: aiogram.types.Message):
    global model
    input_text = message.text.lower()
    response = generate_response(input_text)
    await message.answer(response)

# Запуск бота
if __name__ == '__main__':
    aiogram.executor.start_polling(dp)
