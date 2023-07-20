import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
import numpy as np

# 이미지를 저장할 폴더 경로
folder_path = '/Users/woongjae/Desktop/gradu/musinsa'

# 이미지를 저장할 폴더 생성
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 크롤링할 페이지 범위
start_page = 1
end_page = 20  # 크롤링할 마지막 페이지 번호

product_url = []
product_title = []
product_price = []
product_sale_price =[]

# 페이지 순회하면서 크롤링
for page in range(start_page, end_page + 1):
    # 크롤링할 페이지 URL
    url = f'https://www.musinsa.com/categories/item/001006?d_cat_cd=001006&brand=&list_kind=small&sort=pop_category&sub_sort=&page={page}&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&plusDeliveryYn=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure='

    # 페이지 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상품 리스트 추출
    product_list = soup.find_all('li', attrs={'class':'li_box'})

    # 각 상품 정보 추출
    for product in product_list:
        
        try:
        # 이미지 URL 추출
            tag1 = product.find('img', attrs={'class':'lazyload lazy'})
            img_url = tag1['data-original']
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
        except:
            img_url = 'None'
        # 제품 이름 추출
        name_tag = product.find('a', attrs={'class':'img-block'})
        name = name_tag['title']
        clothes_url = name_tag['href']
        if clothes_url.startswith('//'):
                clothes_url = 'https:' + clothes_url

        # 가격 추출
        price = None
        price_tag = product.find('p', class_='price')
        
        if price_tag:
            try:
                price1 = price_tag.get_text().split()[0]
                #price2 = price_tag.get_text().split()[1]
            except:
                continue


        # 이미지 다운로드
        img_data = requests.get(img_url).content
        img_name = name.replace('/', '_')  # 파일명에 '/' 문자가 포함되면 '_'로 대체
        with open(f'{folder_path}/{img_name}.jpg', 'wb') as f:
            f.write(img_data)
            print(f'{img_name}.jpg 다운로드 완료!')

        # 상품 정보 출력
        print('url :', img_name)
        print('제품명:', name)
        print('정가:', price1)
            #print('할인가격:', price2)
        print('---')
        product_title.append(name)
        product_price.append(price1)
       #product_price.append(price2)
        product_url.append(clothes_url)
    
    df = pd.DataFrame({'상품명 : ':product_title,'정가 : ':product_price, '할인가 :':product_sale_price,'주소 :': product_url})
    df.to_csv('무신사.csv', encoding= 'utf-8')
    time.sleep(1)