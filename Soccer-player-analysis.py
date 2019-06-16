import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

'''设置字体的方法'''
# 1.利用 matplotlib.pyplot 设置
plt.rcParams['font.sans-serif']=['SimHei']
# # 2.利用 matplotlib 设置
# mpl.rcParams['font.family']='SimHei'
# mpl.rcParams['axes.unicode_minus']=False
# # 3.利用 seaborn 设置
# sns.set(style='darkgrid',font='SimHei',font_scale=1.5,rc={"axes.unicode_minus":False})
# warnings.filters('ignore')

'''一、加载数据集'''
# 读取数据（参数指定的文件），返回DataFrame 对象
# data=pd.read_csv('F:\python练习\数据分析\sports-data\data.csv')
# print(data.head())
# print(data.info())

columns = ["Name", "Age", "Nationality", "Overall", "Potential", "Club", "Value", "Wage",
"Preferred Foot",
"Position", "Jersey Number", "Joined", "Height", "Weight", "Crossing", "Finishing",
"HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", "Curve", "FKAccuracy",
"LongPassing",
"BallControl", "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance",
"ShotPower",
"Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions",
"Positioning", "Vision",
"Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving",
"GKHandling",
"GKKicking", "GKPositioning", "GKReflexes", "Release Clause"]

data=pd.read_csv('F:\python练习\数据分析\sports-data\data.csv',usecols=columns)
# print(data.head())
# print(data.info())
# 设置显示最大的列数
pd.set_option('max_columns',50)
# print(data.head())

'''二、数据清洗'''
'''1. 缺失值处理'''
# 方法：① 通过 info 查看数据信息 ； ② 通过 isnull 与 sum 结合，查看缺失值情况
# info方法可以显示每列名称，非空值数量，每列的数据类型，内存占用等信息。
# print(data.info())
# # 查看统计缺失值情况
# print(data.isnull().sum())
# 删除所有含有空行的数据 ， 就地修改
data.dropna(axis=0,inplace=True)
# print(data.isnull().sum())
# print(data.info())

'''2.异常值处理'''
# #通过describe查看数值信息
# print(data.describe())
# # 可配合箱线图辅助。
# # 异常值可以删除，视为缺失值，或者不处理
# sns.boxplot(data=data[["Age","Overall"]])
# plt.show()

'''3.重复值处理'''
# # 使用duplicated 检查重复值， 可配合 keep 参数进行调整
# print(data.duplicated().sum())
# # 使用drop_duplicates 删除重复值
# print(data.drop_duplicates(inplace=True))


'''三、数据分析'''

'''1.数据转换'''
# 注：通过数据分布可知，身高与体重 并不是数值类型， 需要进行转化，然后进行统计
# 拓展：
# 1英尺 = 30.48厘米      1英寸 = 2.54厘米     1磅 = 0.45千克
# print(data.info())
# print(data['Height'].head())
# print(data['Weight'].head())

# 定义转换函数
def change_height(height):
    heights=height.split("'")
    return int(heights[0])*30.48+int(heights[1])*2.54
data['Height']=data['Height'].apply(change_height)
# print(data["Height"])

def change_weight(weight):
    weights=weight.replace("lbs","")
    return int(weights)*0.45
data['Weight']=data['Weight'].apply(change_weight)
# print(data['Weight'])

'''2.绘制 核密度图'''
# # 分析 并绘制身高与 体重的分布
# fig,ax=plt.subplots(1,2)
# fig.set_size_inches((35,20))
# sns.distplot(data[['Weight']],bins=50,ax=ax[0],color='r',axlabel='Weight')
# sns.distplot(data[['Height']],bins=50,ax=ax[1],axlabel='Height')
# # plt.show()

'''3.统计分析'''

'''问题1： 左撇子适合 踢足球吗？？？？'''
# 1)从数量上对比 (惯用脚数)
# number=data['Preferred Foot'].value_counts()
# # print(number)
# # 画图
# sns.countplot(data['Preferred Foot'],data=data)
# # plt.show()
# # 得知：球员在球场上用Left/ Right Foot 的情况

# 2）从能力上对比
# # print(data.info())
# quality=data.groupby('Preferred Foot')['Overall']
# # quality=[i for j in quality for i in j]
# # print(quality)
# sns.boxplot(x='Preferred Foot',y='Overall',data=data)
# plt.show()

# 3）从位置上对比
# # 由于在综合能力上体现不明显， 所以 ，通过每个位置，进行细致的分析；
# # 为了分析的客观性，只统计左脚 与右脚都超过50人（含50人）的位置
# t=data.groupby(['Preferred Foot','Position']).size()
# # 让数据并排显示
# t=t.unstack()
# # 给 人数小于 50人的位置赋予 nan
# t[t<50]=np.nan
# # 删除列中含有 nan 的 列
# t.dropna(axis=1,inplace=True)
# # print(t)
# # 然后， 根据计算的位置， 对数据进行过滤
# t1=data[data['Position'].isin(t.columns)]
# # print(t1)
# plt.figure(figsize=(35,20),dpi=100)
# sns.barplot(x='Position',y='Overall',hue='Preferred Foot',hue_order=['Left','Right'],data=t1)
# plt.xticks(fontsize=25)
# plt.show()
# # 可以得知：左脚选手 更适合 RW（右边峰）的位置

'''问题2： 哪个俱乐部/ 国家拥有综合能力更好的球员 （top10）'''
# 由于人数不一， 为了方便统计， 只考虑人数达到 一定规模的俱乐部/ 国家
'''1.俱乐部'''
# club=data.groupby('Club')
# # print(club)
# overall=club['Overall'].agg(['mean','count'])
# # print(overall)
# # 大于等于 20
# overall=overall[overall['count']>=20]
# overall=overall.sort_values(by='mean',ascending=False).head(10)
# # print(overall.index)
# # 画图
# overall.plot(kind='bar')
# plt.show()
##可以得知：知名俱乐部平均能力更好的 球员，但并非球员平均能力越好，球队的成绩就越好

'''2.国家队'''
# country=data.groupby('Nationality')
# overall=country['Overall'].agg(['mean','count'])
# # print(overall['count'])
# overall=overall[overall['count']>=50]
# # print(overall.head())
# overall=overall.sort_values('mean',ascending=False).head(10)
# overall.plot(kind='bar')
# plt.show()
# print(overall.head())
# # 一些知名足球国家， 在球员的平均能力上可能没有非常靠前，只是因为球员较多，进而个别 球员较知名而已

'''问题3：哪个俱乐部拥有效力更久的球员 （5年以上）'''
# # print(data.head(1))
# # print(data['Joined'])
# # 将时间转化为 数值型时间
# t=pd.to_datetime(data['Joined'])
# t=t.astype(np.str)
# join_year=t.apply(lambda item:int(item.split("-")[0]))
# # print(join_year)
# over_five_years=(2019-join_year)>=5
# # print(data[over_five_years])
# t2=data[over_five_years]
# # print(type(t2))
# t2=t2['Club'].value_counts()
# # print(t2)
# t2.iloc[:15].plot(kind='bar')
# plt.show()

'''问题4：足球运动员是否与出生日期相关'''
# data2=pd.read_csv('F:\python练习\数据分析\sports-data\wc2018-players.csv')
# # print(data2.head())
# # print(data2.info)
# # expand=True 使 数据类型不发生变化
# t=data2['Birth Date'].str.split('.',expand=True)
# print(t.head())
# # print(type(t))
# # 测试 出生
# print(t[0].value_counts())
# # t[0].value_counts().plot(kind='bar')     # 按 出生的 天 进行统计
# # t[1].value_counts().plot(kind='bar')     # 按 出生的 月份 进行统计
# # plt.xlabel('month')
# # plt.ylabel('人数')
# t[2].value_counts().plot(kind='bar')      # 按 出生的 年份 进行统计
# plt.show()
###可以得知：足球运动员与出生日期是 有关的，在年初出生的运动员明显多于在年末出生的运动员

'''问题5：足球运动员号码是否 与 位置相关'''
# # print(data.head(1))
# loction=data.groupby(['Jersey Number','Position'])
# t=loction.size()
# t=t[t>=100]
# # print(t)
# t.plot(kind='bar')
# plt.show()
### 可以得知，足球运动员的号码与 位置是相关的，1号 通常是守门员， 9号是中锋等

'''问题6：身价与薪水、违约金是否相关'''
# 通过数据表可知， 身价与违约金的单位 既有 M ，也有K，故，需要统一 K单位；
# 同时，将数据类型换为 数值类型， 便于统计
# print(data['Value'])      # 身价
# print(data['Wage'])         # 薪水
# print(data['Release Clause'])   # 违约金

# def to_numer(item):
#     item=item.replace('€','')
#     value=float(item[:-1])
#     if item[-1]=='M':
#         value=value*1000
#         return value
#     else:
#         return value
# data['Value']=data['Value'].apply(to_numer)
# data['Wage']=data['Wage'].apply(to_numer)
# data['Release Clause']=data['Release Clause'].apply(to_numer)
# # print(data['Wage'].head)
# # 画图
# sns.scatterplot(x='Value',y='Wage',data=data)     # 身价 与薪水
# sns.scatterplot(x='Value',y='Release Clause',data=data)     # 身价与 违约金
# sns.scatterplot(x='Value',y='Height',data=data)          # 身价与身高
# plt.show()
##可以得知：  足球运动员的身价与其薪水是紧密关联的，尤其是违约金，与身价的关联更大。

'''问题7：哪些指标 对综合评分的影响较大'''
# # 相关系数
# corr=data.corr()
# # print(corr.index)
# # print(corr.shape[0])
# # 画图
# plt.figure(figsize=(40,25),dpi=150)
# sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap=plt.cm.Greens)
# plt.xticks(range(corr.shape[0]),corr.index,fontsize=20,rotation=45)
# plt.yticks(range(corr.shape[1]),corr.index,fontsize=20)
# # 保存图片
# # plt.savefig('corr.png',dpi=200,bbox_inches='tight')
# plt.show()
## 可以得知：Reactions（反应）与Composure（沉着）两项技能对总分的影响最大

'''问题8：分析 某项未标记的技能'''
# # 假设因为某些原因，GKDiving列的标题没有成功获取，现在分析该技能可能表示的含义。
# g=data.groupby('Position')
# g['GKDiving'].mean().sort_values(ascending=False)
# # print(g['GKDiving'].mean())
# plt.figure(figsize=(35,20),dpi=100)
# sns.barplot(x='Position',y='GKDiving',data=data)
# plt.show()
#

'''问题9：年龄与 评分的关系'''
# # sns.scatterplot(x='Age',y='Overall',data=data)
# # plt.show()
#
# # 年龄与 评分的相关系数
# age_overall_corr=data['Age'].corr(data['Overall'])
# # print(age_overall_corr)
# # print(data['Age'])
# # print(data['Age'].min(),data['Age'].max())
#
# min,max=data['Age'].min()-0.5,data['Age'].max()
#
# # # 对一个数组进行切分， 可以将连续值 变成离散值
# # # bins 指定区间数量， bins 如果是 int 类型， 则进行等分
# # # 此处的区间未 左开右闭
# # t=pd.cut(data['Age'],bins=4)
# # t=pd.concat((t,data['Overall']),axis=1)
# # sns.lineplot(y=t['Overall'],markers='*',ms=30,x=data['Age'],data=t)
# # plt.show()
#
# # # 如果需要 进行区间的不等分，则可以将 bins 参数指定为数组类型
# # # 数组来指定区间的 边界；
# # # cut 默认显示的内容 为区间的范围， 如果自定义内容（制定每个区间显示的内容），可通过 labels参数进行指定。
# # t=pd.cut(data['Age'],bins=[min,20,30,40,max],labels=["弱冠之年", "而立之年","不惑之年", "知天命"])
# # t=pd.concat((t,data['Overall']),axis=1)
# # g=t.groupby('Age')
# # # print(g.mean())
# # sns.lineplot(y='Overall',markers='*',ms=30,x='Age',data=t)
# # plt.show()
## 可以得知： 随着年龄的增长，球员得到更多的锻炼与经验，总体能力提升，但三十几岁之后，由于体力限制，总体能力下 降。

'''总结'''
# 1. 左撇子相对于右撇子来说，并无明显劣势，其更适合右边锋的位置。
# 2. 知名俱乐部平均能力更好的球员，但并非球员平均能力越好，球队的成绩就越好。
# 3. 一些知名足球国家，在球员的平均能力上可能并没有非常靠前，只是因为足球运动员较多，进而个别球员较知 名而已。
# 4. 足球运动员与出生日期是有关的，在年初出生的运动员要明显多于在年末出生的运动员。
# 5. 足球运动员的号码与位置是相关的，例如，1号通常都是守门员，9号通常是中锋等。
# 6. 足球运动员的身价与其薪水是紧密关联的，尤其是违约金，与身价的关联更大。
# 7. Reactions（反应）与Composure（沉着）两项技能对总分的影响最大。
# 8. 随着年龄的增长，球员得到更多的锻炼与经验，总体能力提升，但三十几岁之后，由于体力限制，总体能力下 降。

