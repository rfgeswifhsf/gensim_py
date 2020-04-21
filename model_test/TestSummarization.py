#coding=utf-8
'''
通过从文本中提取最重要的句子来演示文本摘要。
'''

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from pprint import pprint
from gensim.summarization import summarize

text = '''Thomas A. Anderson is a man living two lives. By day he is an 
    average computer programmer and by night a hacker known as 
    Neo. Neo has always questioned his reality, but the truth is 
    far beyond his imagination. Neo finds himself targeted by the 
    police when he is contacted by Morpheus, a legendary computer 
    hacker branded a terrorist by the government. Morpheus awakens 
    Neo to the real world, a ravaged wasteland where most of 
    humanity have been captured by a race of machines that live 
    off of the humans's body heat and electrochemical energy and 
    who imprison their minds within an artificial reality known as 
    the Matrix. As a rebel against the machines, Neo must return to 
    the Matrix and confront the agents: super-powerful computer 
    programs devoted to snuffing out Neo and the entire human 
    rebellion. '''
#summarize(text, ratio=0.2, word_count=None, split=False)

pprint(summarize(text))

pprint(summarize(text, split=True)) #If True, list of sentences will be returned
print('\n')
print(summarize(text,ratio=0.1)) #用于确定摘要选自原文的句子，越大摘要越多
print('\n')
print(summarize(text,word_count=50))#确定输出将包含多少单词。
print('\n')


# 关键字提取，基本以名词为主
# keyword 获取所提供文本和/或其组合中排名最靠前的单词
from gensim.summarization import keywords
# keywords(text, ratio=0.2, words=None, split=False, scores=False, pos_filter=('NN', 'JJ'),
#              lemmatize=False, deacc=True)
print('key_words : ',keywords(text,words=5))
print('key_words带权重：',keywords(text,scores=True))
print('key_words_split',keywords(text,split=True))

# mz_keywords 利用Montemurro和Zanette熵算法从文本中提取关键词。
from gensim.summarization import mz_keywords

# mz_keywords(text, blocksize=1024, scores=False, split=False, weighted=True, threshold=0.0)
print('mz',mz_keywords(text,blocksize=10,weighted=False,scores=True,threshold='auto'))
print('mz',mz_keywords(text,blocksize=5,weighted=False,scores=True,threshold=0.01))
