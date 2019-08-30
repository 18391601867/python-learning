''' 抓取 json 文件'''
'''  抓取哪吒数据并储存 csv '''
import pandas as pd
import requests
import time
import os
from fake_useragent import UserAgent

''' 时间转时间戳 '''
def time_stamp(dt):
    timeArray = time.strptime(dt, "%Y-%m-%d") # 字符串 转 时间数组
    timestamp = int(time.mktime(timeArray)) # 时间数组 转 时间戳
    return timestamp
  
headers = {
    'User-Agent': UserAgent().random
}
m = {}
start_time = time_stamp('2019-07-26') # 哪吒上映时间
# 结束时间，可以自定义 ; 或者 已当前时间结束
end_time = int(time.time()) # 以当前日期为结束时间

while True:
    time_local = time.localtime(start_time)    # 时间数组 转 新的时间格式 20190726(url 所需的格式)
    start_time += 86400                        # 加一天
    dt = time.strftime("%Y%m%d",time_local)    # 时间
    print('正在抓取 %s 的数据' %dt)
    url = 'https://box.maoyan.com/promovie/api/box/second.json?beginDate=' + dt
    r = requests.get(url,headers=headers) # 请求接口
    res = r.json()                       # 返回 json 文件
    movie_list = res['data']['list']    # 每日电影列表
    queryDate = res['data']['queryDate']    # 日期
    # print(queryDate)
    if res['success'] and time_stamp(queryDate) <= end_time: # 控制抓取时间

        for movie in movie_list:
            if movie['movieName'] == '哪吒之魔童降世' and movie['releaseInfo'] != '点映':
                m[movie['movieName']] = m.get(movie['movieName'], [])
                m[movie['movieName']].append(movie['boxInfo'])
    else:
        print('结束')
        break

''' 利用pandas 数据储存  '''
nezha_df = pd.DataFrame.from_dict(m, orient='index')
nezha_df = nezha_df.T
nezha_df.to_csv('哪吒.csv', sep=',', encoding='utf_8_sig', index=False)

