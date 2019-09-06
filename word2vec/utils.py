from gensim import models
import pandas as pd
import jieba

train_data = pd.read_csv("../datas/Train/Train_DataSet.csv")
test_data = pd.read_csv("../datas/Test_DataSet.csv")

stopwords = []
with open("../stopwords.txt") as f:
    stopwords = [word.strip() for word in f.readlines()]


class Txt2Word(object):
    def __init__(self, text):
        self.text = text

    def __iter__(self):
        for line in text.values:
            words = jieba.cut(str(line[0]))
            result_word = [word for word in words if word not in stopwords and word != '\n' and word != '']
            yield result_word

text = pd.DataFrame()
text['content'] = train_data['title'] + train_data['content']
text['content'].append(test_data['title'] + test_data['content'])
texts = Txt2Word(text['content'])
model = models.Word2Vec(texts, workers=4, min_count=5, size=100,sg=1, window=5, negative=5)
model.wv.save_word2vec_format("./word.vector", binary=True)