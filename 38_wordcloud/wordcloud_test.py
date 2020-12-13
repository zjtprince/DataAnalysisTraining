from   wordcloud  import WordCloud

import jieba

from matplotlib import pyplot as plt

def generate_wordcloud(text):
    tokenizer = " ".join(jieba.cut(text,cut_all=False, HMM=True,))
    wc = WordCloud(
        font_path='/home/zjtprince/Documents/word_cloud/SimHei.ttf'
        ,max_words=100
        ,width=2000
        ,height=1200
    )

    wc.generate(tokenizer)
    wc.to_file("word_cloud.jpg")
    plt.imshow(wc)
    plt.show()

def remove_stop_words(f):
    stop_words = ['的','什么','学习','如何' ]
    for sw in stop_words:
        f.replace(sw , "")
    return f


words = '数据分析全景图及修炼指南\学习数据挖掘的最佳学习路径是什么？\
    Python基础语法：开始你的Python之旅\
    Python科学计算：NumPy\
    Python科学计算：Pandas\
    学习数据分析要掌握哪些基本概念？\
    用户画像：标签化就是数据的抽象能力\
    数据采集：如何自动化采集数据？\
    数据采集：如何用八爪鱼采集微博上的“D&G”评论？\
    Python爬虫：如何自动化下载王祖贤海报？\
    数据清洗：数据科学家80%时间都花费在了这里？\
    数据集成：这些大号一共20亿粉丝？\
    数据变换：大学成绩要求正态分布合理么？\
    数据可视化：掌握数据领域的万金油技能\
    一次学会Python数据可视化的10种技能'

generate_wordcloud(remove_stop_words( words))

s='是什么什么东西'
print(s.replace('什么',''))