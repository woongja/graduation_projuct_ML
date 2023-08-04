from googleapiclient.discovery import build
import json

my_api_key = "AIzaSyBvuSn26PtTfUeMURIHBa7xSo_xoJxUwoo"
my_cse_id = "d6c7e99caab824b0a"  
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res
#구글로 검색하기
search_result = google_search("Python Google Custom Search", my_api_key, my_cse_id)
#전체 크롤링 결과 파일로 쓰기
with open('search_result_all.json', 'w', encoding='utf-8') as f:
    json.dump(search_result, f, ensure_ascii=False, indent=4)
items=search_result['items']
#query 결과에서 link만 추출해서 print 하기
for i in items:
    print(str(i['link']))