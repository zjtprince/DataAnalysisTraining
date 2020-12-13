import jieba
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

BASE_DIR='/home/zjtprince/Documents/text_classification/text classification/'

def cut_text(filepath):
    text = open(filepath,'r',encoding='gb18030').read()
    words = jieba.cut(text)
    return ' '.join(words) ;

def load_features_and_labels(dir , label):
    features = []
    labels = []
    files = os.listdir(dir)
    for file in files:
        features.append(cut_text(dir + os.sep + file))
        labels.append(label)
    return features , labels

def build_word_list_and_label_list(type_name):
    train_features1, labels1 = load_features_and_labels(BASE_DIR+type_name+'/女性', '女性')
    train_features2, labels2 = load_features_and_labels(BASE_DIR+type_name+'/文学', '文学')
    train_features3, labels3 = load_features_and_labels(BASE_DIR+type_name+'/校园', '校园')
    train_features4, labels4 = load_features_and_labels(BASE_DIR+type_name+'/体育', '体育')
    train_list = train_features1 + train_features2 + train_features3 + train_features4
    label_list = labels1 + labels2 + labels3 + labels4
    return train_list, label_list

def load_stop_words():
    stop_words = open(BASE_DIR+"stop/stopword.txt", 'r',encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')
    return stop_words.split('\n')

if __name__ == '__main__':
    stop_words = load_stop_words()
    train_list, label_list = build_word_list_and_label_list('train')
    test_list, test_labels = build_word_list_and_label_list('test')

    vec = TfidfVectorizer(stop_words=stop_words)
    vec.fit(train_list)
    train_data = vec.transform(train_list)
    test_data = vec.transform(test_list)

    bayes = MultinomialNB(alpha=0.001)
    ctf = bayes.fit(train_data, label_list)

    predict = ctf.predict(test_data)
accur = accuracy_score(predict,test_labels)
print("准确率为：%f" , accur)