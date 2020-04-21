'''
超平面多维近似向量查找工具annoy使用总结

gensimh属于加载到内存中暴力搜索，即全面遍历比价余弦相似度来进行查找。数据量大-->内存压力;搜索慢;
几十万个向量可以直接使用word2vec直接查找。
百万级的向量用annoy内存占用少，速度快。

另一种方法是 Faiss

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
model = word2vec.Word2Vec(sens_list,min_count=1,iter=20,size=5)

from gensim.similarities.index import AnnoyIndexer

'''
对model进行聚类计算
建立一个二叉树集合的索引（树的数量为100），
查找相似
'''
annoy_index = AnnoyIndexer(model,100)
vector = model.wv["烤鸭"]
# The instance of AnnoyIndexer we just created is passed
approximate_neighbors = model.wv.most_similar([vector], topn=3, indexer=annoy_index)
# Neatly print the approximate_neighbors and their corresponding cosine similarity value

print("Approximate Neighbors")
for neighbor in approximate_neighbors:
    print(neighbor)

normal_neighbors = model.wv.most_similar([vector], topn=3)
print("\nNormal (not Annoy-indexed) Neighbors")
for neighbor in normal_neighbors:
    print(neighbor)
