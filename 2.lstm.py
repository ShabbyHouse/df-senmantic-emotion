from keras import Sequential
from keras.engine.saving import model_from_yaml
from keras.layers import Embedding,LSTM,Dropout,Activation,Dense
import keras
from keras_preprocessing import sequence
import yaml
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import numpy as np
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer


class Model(object):
    def __init__(self, pro_name="df"):
        self.max_seq_len = 100
        self.lstm_size = 128
        self.epoch = 10
        self.batch_size = 128
        self.embedding_input_dim = 10000
        self.embedding_output_dim = 128
        self.max_words = 10000
        self.max_len = 100
        self.split = 10

        self.pro_name = pro_name
        self.model_name = "model/%s_seq%d_lstm%d_epochs%d.h5" % (self.pro_name,self.max_seq_len,self.lstm_size,self.epoch)
        self.ymal_name = "model/%s_seq%d_lstm%d_epochs%d.ymal" % (self.pro_name,self.max_seq_len,self.lstm_size,self.epoch)

    def split_data(self,x,y):
        x_train,x_validate,y_train,y_validate = train_test_split(x,y,test_size=0.1)

        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(x_train)
        train_seq = tokenizer.texts_to_sequences(x_train)
        x_train = sequence.pad_sequences(train_seq, maxlen=self.max_len)

        tokenizer.fit_on_texts(x_validate)
        val_seq = tokenizer.texts_to_sequences(x_validate)
        x_validate = sequence.pad_sequences(val_seq, self.max_len)

        y_train = keras.utils.to_categorical(x_validate,num_classes=3)
        y_validate = keras.utils.to_categorical(y_validate,num_classes=3)

        return x_train,y_train,x_validate,y_validate

    def build_model_lstm(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embedding_input_dim,output_dim=self.embedding_output_dim))
        model.add(LSTM(units=self.lstm_size,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        return model

    def train(self,x,y):
        lstm_model = self.build_model_lstm()
        lstm_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

        for i in range(self.epoch):
            x_train, y_train, x_validate, y_validate = self.split_data(x,y)
            lstm_model.fit(x_train,y_train,batch_size=self.batch_size,validation_data=(x_validate,y_validate))

        yaml_string = lstm_model.to_yaml()
        with open(self.ymal_name,'w') as outfile:
            outfile.write(yaml.dump(yaml_string,dafault_flow_style=True))

        lstm_model.save_weights(self.model_name)

    def predict(self,x):
        with open(self.ymal_name) as f:
            yaml_string = yaml.load(f)
        lstm_model = model_from_yaml(yaml_string)

        lstm_model.load_weights(self.model_name,by_name=True)
        lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

        x = sequence.pad_sequences(x,self.max_seq_len)
        predicts = lstm_model.predict_classes(x)

        classes = [0,1,2]
        predicts = [classes[x] for x in predicts]

        return predicts


train_id = []
train_text = []
with open('./datas/Train/Train_DataSet.csv',encoding='utf8') as f:
    for idx,i in enumerate(f):
        if idx == 0:
            pass
        else:
            ik = str(i).split(',')[0]
            i = str(i).split(',')[1:]
            train_id.append(ik)
            train_text.append(','.join(i).replace('\\n',' ').replace('\\r\\n','').replace(' ','').replace('.',''))
train = pd.DataFrame()
train['id'] = train_id
train['text'] = train_text
train_label = pd.read_csv('./datas/Train/Train_DataSet_Label.csv',sep=',')
train = pd.merge(train,train_label,on=['id'],copy=False)

test_id = []
test_text = []
with open('./datas/Test_DataSet.csv',encoding='utf8') as f:
    for idx,i in enumerate(f):
        if idx == 0:
            pass
        else:
            ik = str(i).split(',')[0]
            i = str(i).split(',')[1:]
            test_id.append(ik)
            test_text.append(','.join(i).replace('\\n',' ').replace('\\r\\n','').replace(' ','').replace('.',''))
test = pd.DataFrame()
test['id'] = test_id
test['text'] = test_text
# 合并数据
train_and_test = pd.concat([train,test],ignore_index=True)
# jieba分词
train_and_test['cut_text'] = train_and_test['text'].apply(lambda x:' '.join(jieba.cut(str(x))))
train_shape = train.shape
train_X = train_and_test['cut_text'].values[:train_shape[0]]
train_y = train_and_test['label'][:train_shape[0]]

model = Model()
model.train(train_X,train_y)

test_X = train_and_test['cut_text'].values[train_shape[0]:]

result = pd.DataFrame()
result['id'] = test['id']
result['label'] = model.predict(test_X)
result[['id', 'label']].to_csv('./lstm.csv', index=False)






