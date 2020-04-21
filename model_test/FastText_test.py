'''
fasttext设计的初衷就是为了作为一个文档分类器，副产品是也生成了词向量。
FastText是Facebook在2016年提出的Word2Vec的扩展。fastText的模型架构类似于CBOW
FastText不是将单个词输入神经网络，而是将词分成几个n-gram（sub-words）。

树的结构是根据类标的频数构造的霍夫曼树

虽然训练FastText模型需要更长的时间（n-gram的数量>单词的数量），但它比Word2Vec表现更好，并且允许恰当地表示罕见的单词。

词向量训练以及OOV（out-of-word）问题有效解决
在线更新vocab
oov
'''
from gensim.models import FastText
sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]

model = FastText(sentences,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)
print(model['你'])  # 词向量获得的方式
print(model.wv['你']) # 词向量获得的方式

model.save('FastTest.model')


# 在线更新训练 fasttext


from gensim.models import FastText
sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
sentences_2 = [["dude", "say", "wazzup!"]]

model = FastText(min_count=1,size=5)
model.build_vocab(sentences_1)
model.train(sentences_1, total_examples=model.corpus_count, epochs=model.iter)

model.build_vocab(sentences_2, update=True)
model.train(sentences_2, total_examples=model.corpus_count, epochs=model.iter)

print('cat',model['cat'])
print('dude',model['dude'])

'''
# fasttext自带的OOV功能
# 方式：
# 1 找到每个词的N-grams，_compute_ngrams函数
# 2 然后与n-grams词库进行匹配
# 3 匹配到的n-gram向量平均即为最后的输出值
'''

print('未登录词的向量：',model['hahaha'])

'''

得出的结论：

    具有n-gram的FastText模型在语法任务上的表现明显更好，因为句法问题与单词的形态有关；
    Gensim word2vec和没有n-gram的fastText模型在语义任务上的效果稍好一些，可能是因为语义问题中的单词是独立的单词而且与它们的char-gram无关；
    一般来说，随着语料库大小的增加，模型的性能似乎越来越接近。但是，这可能是由于模型的维度大小保持恒定在100，而大型语料库较大维度的模型大小可能会导致更高的性能提升。
    随着语料库大小的增加，所有模型的语义准确性显着增加。
    然而，由于n-gram FastText模型的语料库大小的增加，句法准确度的提高较低（相对和绝对术语）。这可能表明，在较大的语料库大小的情况下，通过合并形态学信息获得的优势可能不那么显着（原始论文中使用的语料库似乎也表明了这一点）


'''
