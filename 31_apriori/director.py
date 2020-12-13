
from selenium import webdriver
from lxml import etree
import time
import csv
import pandas as pd
import fptools as fp
from efficient_apriori import apriori
director = '张艺谋'
file_name = director + ".csv"

def fetch_data(file_name):
    already_deal_movies = set()

    out = open(file_name, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(out)
    for i in range(0, 151, 15):
        url = 'https://search.douban.com/movie/subject_search?search_text=' + director + '&cat=1002&start=' + str(i)
        driver = webdriver.Chrome()
        driver.get(url)
        time.sleep(1)
        html = etree.HTML(driver.page_source)
        movie_name_pattern = "//div[@class='item-root']/div[@class='detail']/div[@class='title']/a"
        movie_actors_pattern = "//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']"

        movie_list = html.xpath(movie_name_pattern)
        actor_list = html.xpath(movie_actors_pattern)
        movie_count = len(movie_list)
        if (movie_count == 0): break
        if (i == 0):
            movie_list = movie_list[1:]
            actor_list = actor_list[1:]
        for movie, actors in zip(movie_list, actor_list):
            if (actors.text is None or movie.text in already_deal_movies): continue
            actor = actors.text.split('/')
            if (actor[0].strip() != director): continue
            actor[0] = movie.text.strip()
            csv_writer.writerow(actor)
            already_deal_movies.add(movie.text)
        driver.close()
    out.close()

#fetch_data(file_name)

lists = csv.reader(open(file_name, 'r', encoding='utf-8'))

# 数据加载
data = []
for names in lists:
    name_new = []
    for name in names:
        # 去掉演员数据中的空格
        name_new.append(name.strip())
    data.append(name_new[1:])
# 挖掘频繁项集和关联规则
# print(data)
itemsets, rules = apriori(data, min_support=0.05)
print(itemsets)
print(rules)

