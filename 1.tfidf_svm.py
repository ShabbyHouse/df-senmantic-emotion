#coding:utf-8
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold,StratifiedKFold
import re
import warnings
warnings.filterwarnings('ignore')
# 传统方案 TFIFD + lr
# 1 读取数据
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
print(train_label.shape,train.shape,test.shape)
# 合并数据
train_and_test = pd.concat([train,test],ignore_index=True)
# jieba分词
train_and_test['cut_text'] = train_and_test['text'].apply(lambda x:' '.join(jieba.cut(str(x))))
# 几折交叉验证
N = 5
# tifldf特征提取
train_shape = train.shape
# print('make_feat')
tf = TfidfVectorizer(ngram_range=(1,4),analyzer='char')
tf_feat = tf.fit_transform(train_and_test['cut_text'].values)
tf_feat = tf_feat.tocsr()
X = tf_feat[:train_shape[0]]
y = train_and_test['label'][:train_shape[0]]

sub = tf_feat[train_shape[0]:]

kf = StratifiedKFold(n_splits=N,random_state=42,shuffle=True)
oof = np.zeros((X.shape[0],3))
oof_sub = np.zeros((sub.shape[0],3))
for j,(train_in,test_in) in enumerate(kf.split(X,y)):
    print('running',j)
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    clf = SVC(probability=True)
    clf.fit(X_train,y_train)
    test_y = clf.predict_proba(X_test)
    print('shape of test_y:', test_y.shape)
    print('shape of oof:', oof.shape)
    oof[test_in] = test_y
    print('shape of oof_sub:', oof_sub.shape)
    oof_sub = oof_sub + clf.predict_proba(sub)

xx_cv = f1_score(y,np.argmax(oof,axis=1),average='macro')
print(xx_cv)

result = pd.DataFrame()
result['id'] = test['id']
result['label'] = np.argmax(oof_sub,axis=1)
print('finish')
result[['id','label']].to_csv('./baseline_svm_tfidf_{}.csv'.format(str(np.mean(xx_cv)).split('.')[1]),index=False)