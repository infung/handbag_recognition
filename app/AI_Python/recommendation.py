from cmath import nan
import pymysql
import pandas as pd
import logging
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def stem(text):    
    y =[]
    ps = PorterStemmer()
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

def check(x):
    if x == nan:
        print(x)

def recommend(bag):
    if os.path.exists('similarity.pkl') and os.path.exists('masterBagData.pkl'):
        # load similarity matrix from pickle file
        with open('similarity.pkl', 'rb') as f1:
            similarity = pickle.load(f1)
        with open('masterBagData.pkl', 'rb') as f2:
            data = pickle.load(f2)
        logger.info("files exist")
        index = data[data['model'] == bag].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        res = []
        for i in distances[1:10]:
            res.append(data.iloc[i[0]].masterBagId)
        return res

    cv = CountVectorizer(max_features=500,stop_words='english')    
    # 创建链接
    conn = pymysql.connect(
    host='34.92.161.10',  # 本地回环地址
    port=3306,        # mysql固定端口好3306
    user='root',      # 用户名
    password='6CI|iI{p^Xe0/pi<',   # 密码
    database='bagTracker',   # 必须指定库
    charset='utf8'    # 注意这里不可以写utf-8
    )

    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 括号内的命令是让数据自动组织成字典
    # 定义sql语句
    sql = 'select masterBagId, brand, model, category, color, material from MasterBag'
    # 执行sql语句
    cursor.execute(sql)
    # 获取返回结果
    res = cursor.fetchall()

    df = pd.DataFrame(res)
    # delete duplicates
    data = df.copy()
    data = data.drop_duplicates()
    # fill nan value
    data.fillna("nan",inplace=True)

    data['tags'] = data['brand'] + " " + data['model'] + " " + data['category'] + " " + data['color'] + " " + data['material']
    data['tags'] = data['tags'].apply(stem)
    # transform tag
    vector = cv.fit_transform(data['tags']).toarray()
    # similarity matrix
    similarity = cosine_similarity(vector)
    
    # dump similarity matrix and master bag data to local path using pickle
    with open('similarity.pkl', 'wb') as f1:
        pickle.dump(similarity, f1)
    with open('masterBagData.pkl', 'rb') as f2:
        pickle.dump(data, f2)

    index = data[data['model'] == bag].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    res = []
    for i in distances[1:10]:
        res.append(data.iloc[i[0]].masterBagId)
        # print(data.iloc[i[0]].masterBagId, data.iloc[i[0]].model, data.iloc[i[0]].color, sep=' ')
    return res
        
print(recommend("kelly"))