import keras
import pickle
from keras import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import *
from keras.preprocessing import text
from keras_preprocessing import sequence
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from JoinAttLayer import Attention
import numpy as np
import pandas as pd
from BiGRU import BiGRU
from CNN import CNN

from gensim.models.keyedvectors import KeyedVectors

w2v_path = "./word2vec/word.vector"
maxlen = 1200
model_name = "model_cnn_%s.hdf5"
batch_size = 128
epochs = 15


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = list(np.argmax(self.model.predict(self.validation_data[0]), axis=1))
        val_targ = list(np.argmax(self.validation_data[1],axis=1))
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("val_f1:", _val_f1, "val_precision:", _val_precision, "val_recall:", _val_recall, "max f1:", max(self.val_f1s))
        return


def generate_embeddings(tokenizer):
    word_index = tokenizer.word_index
    w2_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True, encoding='utf8', unicode_errors='ignore')
    embeddings_matrix = np.zeros((len(word_index)+1, w2_model.vector_size))
    for word, i in word_index.items():
        if word in w2_model:
            embedding_vector = w2_model[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    return word_index,embeddings_matrix


def generate_tokenizer(data,pickle_path):
    tokenizer = text.Tokenizer(num_words=None)
    tokenizer.fit_on_texts(data)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def train(tokenizer, model_dir):

    word_index, embeddings_matrix = generate_embeddings(tokenizer)

    x_train, x_validate, y_train, y_validate = train_test_split(data['content'], data['label'], test_size=0.1)

    list_tokenized_train = tokenizer.texts_to_sequences(x_train)
    input_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

    list_tokenized_validation = tokenizer.texts_to_sequences(x_validate)
    input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)

    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_validate = keras.utils.to_categorical(y_validate, num_classes=3)

    model1 = CNN().model(embeddings_matrix, maxlen, word_index)
    file_path = model_dir + model_name % "{epoch:02d}"
    checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
    metrics = Metrics()
    callbacks_list = [checkpoint, metrics]
    model1.fit(input_train, y_train, batch_size=batch_size, epochs=epochs,
               validation_data=(input_validation, y_validate), callbacks=callbacks_list, verbose=2)
    del model1


def predict(test_word_path, pickle_path, model_dir, model_index):
    with open(pickle_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        word_index, embeddings_matrix = generate_embeddings(tokenizer)

        model1 = CNN().model(embeddings_matrix, maxlen, word_index)
        model1.load_weights(model_dir + "model_bigru_01.hdf5")

        test_data = pd.read_csv(test_word_path)
        tokenizer.fit_on_texts(test_data["content"].values)
        list_tokenized_test = tokenizer.texts_to_sequences(test_data["content"])
        input_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

        result = pd.DataFrame()
        result['id'] = test_data['id']
        result['label'] = np.argmax(model1.predict(input_test), axis=1)
        result[['id', 'label']].to_csv('./bigru.csv', index=False)


if __name__ == "__main__":

    bigru_model_dir = "./bigru_models/"
    cnn_model_dir = "./cnn_models/"
    pickle_path = "tokenizer_word.pickle"

    train_word_path = "./datas/Train/train_word.csv"
    test_word_path = "./datas/test_word.csv"

    data = pd.DataFrame()
    train_data = pd.read_csv(train_word_path)
    train_data = train_data.apply(lambda x: eval(x[2]), axis=1)

    data = train_data.values

    test_data = pd.read_csv(test_word_path)
    test_data['content'] = test_data.apply(lambda x: eval(x[1]), axis=1)
    data['content'].append(test_data.values)

    tokenizer = generate_tokenizer(data['content'], pickle_path)

    train(tokenizer, cnn_model_dir)
    predict(test_word_path, pickle_path, cnn_model_dir, 4)