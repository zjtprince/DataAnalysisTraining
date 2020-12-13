from selenium import webdriver
from lxml import etree
import requests_test

url ='https://search.douban.com/movie/subject_search?search_text=%E5%90%B4%E4%BA%AC&cat=1002&start=0'

driver = webdriver.Chrome()
driver.get(url)

# print (driver.page_source)
html = etree.HTML(driver.page_source)

pic_pattern = "//div[@class='item-root']/a[@class='cover-link']/img[@class='cover']/@src"
name_pattern = "//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']"

srcs  = html.xpath(pic_pattern)
names = html.xpath(name_pattern)
print(srcs)
print(names)

for src , name in zip(srcs,names):
    print(src, name.text)
    requests_test.download_pic(src,name.text )

