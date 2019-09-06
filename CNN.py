import keras
from keras import Model
from keras.layers import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5


class CNN(object):
    def model(self, embeddings_matrix, maxlen, word_index):
        input = Input(shape=(maxlen,))
        embedding = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(input)

        embedding = Reshape((maxlen, embeddings_matrix.shape[1], 1))(embedding)
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embeddings_matrix.shape[1]), padding='valid',
                        kernel_initializer='normal', activation='relu')(embedding)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embeddings_matrix.shape[1]), padding='valid',
                        kernel_initializer='normal', activation='relu')(embedding)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embeddings_matrix.shape[1]), padding='valid',
                        kernel_initializer='normal', activation='relu')(embedding)

        maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
            conv_0)
        maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
            conv_1)
        maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
            conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=3, activation='softmax')(dropout)

        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)
        model = Model(inputs=input, outputs=output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['acc'])
        return model
