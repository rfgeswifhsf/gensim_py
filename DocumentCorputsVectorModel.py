'''
介绍：语料库，语料，向量，模型,词典
'''
from  pprint import pprint

documents = "Human machine interface for lab abc computer applications"

text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

from collections import defaultdict
stoplist=set('for a of the and to in'.split())
texts=[[word for word in document.lower().split() if word not  in stoplist] for document in text_corpus]
# pprint(texts)
print('----------')

#计算词频
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1

# 过滤低频词
processed_corpus = [[token for token in text if frequency[token]>1 ]for text in texts]
# pprint( processed_corpus)

# 生成词袋
from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
pprint(list(dictionary.items()))
pprint(dictionary.token2id)

# 语料中的词对应得词典id 及频率
new_doc = "Human Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print('语料转成（id，词频）',end='')
pprint(new_vec)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print('语料词袋化',end='')
pprint(bow_corpus)


# TF-IDF
from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "system minors".lower().split()
query_bow = dictionary.doc2bow(words)
print('语料TF-idf',end='')
pprint(tfidf[query_bow])


# 相似度
from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
sims = index[tfidf[query_bow]]
print('相识度',sims)

print('与第{0}篇最相似,该文章内容为:    {1}，相识度为{2}'.format(list(sims).index(max(sims))+1,text_corpus[list(sims).index(max(sims))],max(sims)))
print('\n')
print('语料与语料库各个语料相似度',end='')
pprint(list(enumerate(sims)))




