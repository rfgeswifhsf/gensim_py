'''
Annoy库具有一项有用的功能，可以从磁盘对内存进行索引映射。 当多个进程使用相同的索引时，它将节省内存。
持久化磁盘索引
'''

LOGS = False
if LOGS:
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

from gensim.similarities.index import AnnoyIndexer

annoy_index = AnnoyIndexer(model,100)
vector = model.wv["烤鸭"]
# The instance of AnnoyIndexer we just created is passed
approximate_neighbors = model.wv.most_similar([vector], topn=11, indexer=annoy_index)
# Neatly print the approximate_neighbors and their corresponding cosine similarity values
print("Approximate Neighbors")
for neighbor in approximate_neighbors:
    print(neighbor)

normal_neighbors = model.wv.most_similar([vector], topn=11)
print("\nNormal (not Annoy-indexed) Neighbors")
for neighbor in normal_neighbors:
    print(neighbor)
