
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
documents  = [
    'this is the bayes document',
    'this is the second second document',
    'and the third one',
    'is this the document'
]
stop_words = ['and','the','is']
tfidf_vec = TfidfVectorizer(stop_words=stop_words,max_df=0.5)

matrix = tfidf_vec.fit_transform(documents)

print ("不重复的：", tfidf_vec.get_feature_names())
print("每个单词的ID：",tfidf_vec.vocabulary_)


print(matrix)


print('-'*50)

doc1 = '在中文文档中，最常用的是 jieba 包。jieba 包中包含了中文的停用词 stop words 和分词方法。'
doc2 = '我们需要自己读取停用词表文件，从网上可以找到中文常用的停用词保存在 stop_words.txt，然后利用 Python 的文件读取函数读取文件，保存在 stop_words 数组中。'
doc3 = '直接创建 TfidfVectorizer 类，然后使用 fit_transform 方法进行拟合，得到 TF-IDF 特征空间 features，你可以理解为选出来的分词就是特征。' \
       '我们计算这些特征在文档上的特征向量，得到特征空间 features。'
doc4 = '我们将特征训练集的特征空间 train_features，以及训练集对应的分类 train_labels 传递给贝叶斯分类器 clf，它会自动生成一个符合特征空间和对应分类的分类器。'

documents = [doc1,doc2,doc3,doc4]
# print(documents)
sentence_in_words = [list(jieba.cut(doc)) for doc in documents]

documents = [' '.join( word ) for word in  sentence_in_words]
print(documents)

tfidf_vec = TfidfVectorizer(stop_words=stop_words)
matrix = tfidf_vec.fit_transform(documents)
print ("不重复的单词：", tfidf_vec.get_feature_names())
print("每个单词的ID：",tfidf_vec.vocabulary_)
print(matrix)


nb = MultinomialNB()

a = [1]
b = [2]
c = b + a
print(c)