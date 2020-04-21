import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = [
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

from pprint import pprint
from collections import defaultdict

stop_list=set('for a of the and to in'.split())
texts=[
    [word for word in document.lower().split()if word not in stop_list]
    for document in documents
]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
texts=[
    [token for token in text if frequency[token]>1]
    for text in texts
]
pprint(texts)

from gensim import corpora
dictionary = corpora.Dictionary(texts)
pprint(dictionary.token2id)
dictionary.save('./diction.dict')

new_doc =  "Human computer interaction"

#  def doc2bow(self, document, allow_update=False, return_missing=False):
#  document : list of str
new_vec = dictionary.doc2bow(new_doc.lower().split())
print('new_vec :')
pprint(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
print('corpus : ')
pprint(corpus)
