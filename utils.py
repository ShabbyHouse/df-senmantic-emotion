import numpy as np
import pandas as pd
import jieba

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[3,4,5],[1,1,1]])
b = a + b
c = pd.DataFrame(a, index=['a','b'], columns = ['aa','bb','cc'])
d = pd.DataFrame(b, index=['b','c'],columns=['cc','dd','ee'])
print('pd.concat([c,d]) : \n')
print(pd.concat([c,d]))
