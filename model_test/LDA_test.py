import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as pd
import pandas as pd
from pprint import  pprint
import jieba
from gensim import corpora, models, similarities

data = pd.read_csv("data_train.csv",sep="\t",encoding='utf-8',header=None)
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
print(sens_list)
dictionary = corpora.Dictionary(sens_list)

#  def doc2bow(self, document, allow_update=False, return_missing=False):
#  document : list of str
corpus = [dictionary.doc2bow(texts) for texts in sens_list]
pprint(corpus)

lda = models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=5)

topic_list = lda.top_topics(corpus)
for i, topic in enumerate(topic_list):
    print(i,topic)

# Get a single topic as a formatted string  (topic_id,topic_nums)
print(lda.print_topic(3,10))


# 各个document的主题分布
every_corpus = lda.get_document_topics(corpus)
print(every_corpus)
for i in range(len(corpus)):
    print(every_corpus[i])


# 查看指定document的主题分布
sentences= '在美国大选期间，希拉里的邮件被泄露出来了'
sen_doc=dictionary.doc2bow(jieba.lcut(sentences))

one_corpus = lda.get_document_topics(sen_doc)
print('sentences: ',one_corpus)
