import sys
import pandas as pd
import jieba


class PreProcess(object):
    def __init__(self,file_name,pro_name='df'):
        self.file_name = file_name
        self.pro_name = pro_name
        self.embedding_dim = 256

    def read_csv_file(self):
        sys.reload(sys)
        sys.setdefaultencoding('utf-8')

        data = pd.read_csv(self.file_name,sep=',')
        x = data.content.values
        y = data[self.pro_name].values

        return x,y

    def clean_stop_words(self,sentences):
        stop_words = None
        with open('./stop_words.txt','r') as f:
            stop_words = f.readlines()
            stop_words = [line.replace("\n","") for line in stop_words]

        for i,line in enumerate(sentences):
            for word in stop_words:
                if word in line:
                    line = line.replace(word,"")
            sentences[i] = line

        return sentences

    def get_words_after_jieba(self,sentences):
        words_after_jieba = [[word for word in jieba.cut(line) if word.strip()] for line in sentences]
        return words_after_jieba