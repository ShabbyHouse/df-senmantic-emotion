import pandas as pd
import jieba

stopwords = []
with open("./stopwords.txt") as f:
    stopwords = [word.strip() for word in f.readlines()]


def seg_words(text):
    segged_data = []
    for index, line in enumerate(text.values):
        words = jieba.cut(str(line[1]))
        result_word = [word for word in words if word not in stopwords and word != '\n' and word != ' ']
        segged_data.append([line[0], result_word])
    segged_df = pd.DataFrame(data=segged_data, columns=('id', 'content'))
    return segged_df


def train_process(data_path,label_path,word_path):
    train_data = pd.read_csv(data_path)

    text = pd.DataFrame()
    text['id'] = train_data['id']
    text['content'] = train_data['title'] + train_data['content']
    segged_words = seg_words(text)
    train_label = pd.read_csv(label_path, sep=',')
    train = pd.merge(segged_words, train_label, on=['id'], copy=False)
    train.to_csv(word_path, index=False)


def test_process(data_path,word_path):
    test_data = pd.read_csv(data_path)

    text = pd.DataFrame()
    text['id'] = test_data['id']
    text['content'] = test_data['title'] + test_data['content']
    segged_words = seg_words(text)
    segged_words.to_csv(word_path, index=False)

if __name__ == '__main__':
    #  处理train数据
    train_data_path = './datas/Train/Train_DataSet.csv'
    train_label_path = './datas/Train/Train_DataSet_Label.csv'
    train_word_path = './datas/Train/train_word.csv'

    train_data_path_sample_10 = './datas/Train/Train_DataSet_sample_10.csv'
    train_label_path_sample_10 = './datas/Train/Train_DataSet_Label_sample_10.csv'
    train_word_path_sample_10 = './datas/Train/train_word_sample_10.csv'
    # train_process(train_data_path,train_label_path,train_word_path)

    # 处理test数据
    test_data_path = './datas/Test_DataSet.csv'
    test_word_path = './datas/test_word.csv'

    test_process(test_data_path,test_word_path)