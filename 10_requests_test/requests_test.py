
import requests as req
import json
from lxml import etree

from selenium import webdriver 

def download_pic(url,id):
    filename = "./" + str(id) + ".webp"
    try:
        response = req.get(url, timeout= 10)
        f = open(filename,'wb')
        f.write(response.content)
        f.close()
    except req.exceptions.ConnectionError:
        print("无法下载图片")



def get_json():
    for start in range(0, 40, 20):
        url = "https://www.douban.com/j/search_photo?q='王祖贤'&limit=20&tart="+str(start)
        driver = webdriver.Chrome()
        driver.get(url)
        print(driver.page_source)
        html = etree.HTML(driver.page_source)
        src_xpath = "//div[@class='item-root']/a[@class='cover-link']/img[@class='cover']/@src"
        srcs = html.xpath(src_xpath)
        for s in srcs:
            print(s)
        #jsonobj = json.loads(response.text)
        #
        #for pic in jsonobj['images']:
        #    print(pic['src'])

#get_json()


def parse_url():
    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36'}

    query = '王祖贤'
    url = 'https://www.douban.com/j/search_photo?q='+query+'&limit=20&start=0'
    html = req.get(url,headers=headers).text # 得到返回结果
    print('html:'+html)
    jsonObj = json.loads(html)

    images = jsonObj['images']

    for image in images:
        print(image['src'], image['id'])
        download_pic(image['src'], image['id'])


if __name__ == 'main':
    parse_url()
