import jieba
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import pandas as pd
from  pprint import pprint

pprint(common_texts)

data = pd.read_csv("data_train.csv",sep="\t",encoding='utf-8',header=None)


def segment_sen(sen):
    sen_list = []
    try:
        sen_list = jieba.lcut(sen)
    except:
            pass
    return sen_list
data = pd.read_csv("data_train.csv",sep="\t",encoding='utf-8',header=None)
sentance  = list(data[0])
sens_list = [segment_sen(i) for i in sentance]

'''
[TaggedDocument(words=['烤鸭', '还是', '不错', '的', '别的', '菜', '没什么', '特殊', '的'], tags=[0]),
 TaggedDocument(words=['使用', '说明', '看不懂', '不会', '用', '，', '很多', '操作', '没', '详细', '标明'], tags=[1]),
 TaggedDocument(words=['皇帝', '是', '一个', '国家', '最有', '权势', '的', '男人', '，', '而', '一个', '国家', '最有', '权势', '的', '女人', '是', '太后', '。'], tags=[2])]
'''

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sens_list)]

# 训练模型
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# 保存模型
# from gensim.test.utils import get_tmpfile
# fname = get_tmpfile("my_doc2vec_model")
model.save('doc2vec.model')

model = Doc2Vec.load('doc2vec.model')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

vector = model.infer_vector(["system", "response"]) #array([-0.04324552, -0.03536568,  0.05885698,  0.01039772,  0.05310103],dtype=float32)
pprint(vector)

for i in sens_list:
    pprint(model.infer_vector(i))
# array([ 0.00737118, -0.06142687,  0.02004911,  0.00863192,  0.07555129],
#       dtype=float32)
# array([ 0.04775462,  0.05402783,  0.06107682, -0.08609992,  0.07155881],
#       dtype=float32)
# array([-0.03365721, -0.06552761, -0.07865686, -0.03996021, -0.08913099],
#       dtype=float32)
# array([-0.02090582,  0.01097   , -0.02730564, -0.01093301,  0.03408771],
#       dtype=float32)

sim1=model.similarity('男人','皇帝')
print('两个词相似度',sim1)

sim2=model.n_similarity(sens_list[0],sens_list[1])
print('两组词的相似度(两个语料)',sim2)

sim3=model.most_similar(positive=['男人','皇帝'],negative=['女人'],topn=1)
print(sim3)
# model.doesnt_match
# reset_from(other_model) 从另一个（可能是预先训练过的）模型复制可共享的数据结构。
