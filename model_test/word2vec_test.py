from re import split

import gensim
from gensim.models import fasttext, Word2Vec
from gensim.models import word2vec
import pandas as pd
import logging
import jieba
from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data = pd.read_csv("data_train.csv",sep="\t",encoding='utf-8',header=None)
print(data)
sentance  = list(data[0])
'''
对句子进行分词分词
'''
def segment_sen(sen):
    sen_list = []
    try:
        sen_list = jieba.lcut(sen)
    except:
            pass
    return sen_list
'''
 将数据变成gensim中 word2wec函数的数据格式
'''
sens_list = [segment_sen(i) for i in sentance]
pprint(sens_list)

model = word2vec.Word2Vec(sens_list,min_count=1,iter=20,size=5)

'''
模型保存
'''
model.save("word2vec.model")
'''
查看某个词的词向量
'''
print(model['烤鸭'])


'''词典以外的词无法查看词向量，并会报错  
-- "word '哈哈' not in vocabulary"
 判断词是否在字典中 if word in model
'''
# print(model['哈哈'])

for lword in sens_list:
    for word in lword:
        if word in model:
            print(word,'-----',model[word])

'''
相似词汇
'''
sw=model.most_similar(positive=['男人','皇帝'],negative=['女人'],topn=1)
print(sw)

sim=model.similarity('男人','皇帝')
print('二者相似度为： ',sim)

print('以下哪项不属于该序列：',model.doesnt_match(['男人', '女人', '皇帝', '皇太后', '烤鸭']))

sim1=model.most_similar(['男人'])
print(sim1)

