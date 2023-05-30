from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from keras import layers
from keras import models

import numpy as np
import random
import io

import matplotlib.pyplot as plt

import h5py
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

app = Flask(__name__)

initialized = False

maxlen = 10
words = None
word_indices = None
model_new = None
indices_word = None
x = 0
y = 0
batch_size = 128

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5050'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    return response


def initialize():
    global words, maxlen, word_indices, model_new, indices_word, x, y, batch_size

    current_directory = os.path.dirname(__file__)

    file_path = os.path.join(current_directory, "movie_synopsis.csv")

    with open(file_path, encoding="utf-8") as f:
        text = f.read().lower()

    def ok(x):
        return x.isalpha() | (x == ' ')

    text = "".join(c for c in text if ok(c))
    text = text[:100000]

    # создаем словари для кодирования и декодирования текста
    words = sorted(set(text.split()))
    print("Total words:", len(words))
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # Размер начальной последовательности можно отрегулировать
    sentences = []
    next_words = []
    pointer = 0
    text_list = text.split()
    while pointer < len(words) - maxlen:
        sentences.append(text_list[pointer: pointer + maxlen])
        next_words.append(text_list[pointer + maxlen])
        pointer += 1

    x = np.zeros((len(sentences), maxlen, len(words)))
    y = np.zeros((len(sentences), len(words)))
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x[i, t, word_indices[word]] = 1
        y[i, word_indices[next_words[i]]] = 1

    current_directory = os.path.dirname(__file__)

    file_path = os.path.join(current_directory, "model.hdf5")

    model_new = keras.models.load_model(file_path)


@app.before_request
def before_request():
    global initialized
    if not initialized:
        initialize()
        initialized = True


@app.route('/generate', methods=['POST'])
def complete_text():
    result = ''

    if request.method == 'POST':
        data = request.get_json()
        if data is not None and 'text' in data:
            input_text = data['text']

        # -----------Нейронка-----------------

        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        diversity = 0.5
        print("...Diversity:", diversity)

        generated = ""
        sentence = input_text.lower().split()
        print('...Generating with seed: "' + " ".join(sentence) + '"')

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(words)))

            for t, word in enumerate(sentence):
                if word in word_indices:
                    x_pred[0, t, word_indices[word]] = 1.0
                else:
                    random_word = random.choice(list(word_indices.values()))
                    x_pred[0, t, random_word] = 1.0

            preds = model_new.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            sentence = sentence[1:]
            sentence.append(next_word)
            generated += next_word
            generated += " "
        print("...Generated: ", generated)
        print()

        result = generated

    return jsonify({
        'text': result
    })


if __name__ == '__main__':
    app.run()
