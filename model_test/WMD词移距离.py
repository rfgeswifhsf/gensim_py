'''
    效果出色：充分利用了word2vec的领域迁移能力
    无监督：不依赖标注数据，没有冷启动问题
    模型简单：仅需要词向量的结果作为输入，没有任何超参数
    可解释性：将问题转化成线性规划，有全局最优解
    灵活性：可以人为干预词的重要性

就bag of words而言，有如下缺点：1.没有考虑到单词的顺序，2.忽略了单词的语义信息。因此这种方法对于短文本效果很差，对于长文本效果一般，通常在科研中用来做baseline。

average word vectors就是简单的对句子中的所有词向量取平均。是一种简单有效的方法，但缺点也是没有考虑到单词的顺序

tfidf-weighting word vectors是指对句子中的所有词向量根据tfidf权重加权求和，是常用的一种计算sentence embedding的方法，在某些问题上表现很好，相比于简单的对所有词向量求平均，考虑到了tfidf权重，因此句子中更重要的词占得比重就更大。但缺点也是没有考虑到单词的顺序

'''

import jieba
from gensim.similarities import WmdSimilarity

print(20 * '*', 'loading data', 20 * '*')
f = open('全唐诗.txt', encoding='utf-8')
lines = f.readlines()

words = []
documents = []
useless = [',','　','.', '(', ')', '!', '?', '\'', '\"', ':', '<', '>',
           '，', '。', '（', '）', '！', '？', '’', '“', '：', '《', '》', '[', ']', '【', '】']
for each in lines:
    each = each.replace('\n', '')
    each.replace('-', '')
    each = each.strip()
    each = each.replace(' ', '')
    if (len(each) > 3):
        if (each[0] != '卷'):
            documents.append(each)
            each = jieba.lcut(each)
            text = [w for w in each if not w in useless]
            words.append(text)

print(len(words))

print(20 * '*', 'trainning models', 40 * '*')
from gensim.models import Word2Vec

model = Word2Vec(words, workers=3, size=100)

# Initialize WmdSimilarity.
num_best = 10 #检索数
instance = WmdSimilarity(words, model, num_best=10) #  :class:`numpy.ndarray`,Similarity matrix.--->(idex,sim)

print(20 * '*', 'testing', 40 * '*')
while True:
    sent = input('输入查询语句： ')
    sent_w = jieba.lcut(sent)
    query = [w for w in sent_w if not w in useless]

    sims = instance[query]  # A query is simply a "look-up" in the similarity class.
    # Print the query and the retrieved documents, together with their similarities.
    print('Query: ',sent)
    if len(sims)>0:
        for i in range(num_best):
            print('sim = %.4f' % sims[i][1])
            print(documents[sims[i][0]])
    else:
        print('No similar verse')
