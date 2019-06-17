'''
AQI:指空气质量指数用来衡量空气清洁或污染的程度。值越小，表示空气质量越好。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''设置字体'''
plt.rcParams['font.sans-serif']=['SimHei']

'''一、加载数据项'''
data=pd.read_csv('F:\python练习\数据分析\AQI-data\CompletedDataset.csv')
#设置展示最大的列
pd.set_option('max_columns',20)
# print(data.head())
# print(data.info())
# print(data.shape)
# print(data.sample())

'''二、数据清洗'''

'''1.缺失值处理'''
# print(data.info())
# # 统计数据缺失的部分
# print(data.isnull().sum(axis=0))

'''2.异常值处理'''
# # 通过describe查看数值信息
# print(data.describe())
# # 可配合箱线图 辅助
# # 异常值可以删除，视为缺失值，或者不处理
# sns.boxplot(data=data['Precipitation'])    # 降雨量
# plt.show()

'''3.重复值处理'''
# 使用 duplicated 检查重复值；   # 使用drop_duplicates 删除重复值
# data_dup=data.duplicated().sum()
# print(data_dup)

'''三、数据分析'''
'''问题1：空气质量最好/ 最差的五个城市'''
# 用途：空气质量的好坏可以为我们以后找工作、旅游等提供参考。
'''1.最好的五个城市'''
# # print(data.info())
# city_aqi=data[['City','AQI']].sort_values('AQI')
# # print(city_aqi)
# # 方法一：
# best_five_city=city_aqi.iloc[:5]
# # print(best_five_city)
# # # 方法二：
# # print(city_aqi.head())
# # 画图
# sns.barplot(x='City',y='AQI',data=best_five_city)
# plt.show()
## 可以得知：空气质量最好的五个城市：1. 韶关市 2. 南平市 3. 梅州市 4. 基隆市 5. 三明市

'''2.最差的五个城市'''
# city=data[['City','AQI']].sort_values('AQI',ascending=False)
# worst_five_city=city.iloc[:5]
# # print(worst_five_city)
# sns.barplot(x='City',y='AQI',data=worst_five_city)
# plt.show()
## 可以得知：空气质量最差的五个城市：1. 北京市 2. 朝阳市 3. 保定市 4. 锦州市 5. 焦作市

'''问题2：临海城市空气质量是否优于内陆城市'''
# 首先统计临海与内陆的城市数量
# print(data.head(2))
# 方法一：
# city_number=data['Coastal'].value_counts()
# print(city_number.iloc[:1])      # 非临海
# print(city_number.iloc[1:2])     # 临海
# 方法二：
# 画图
# sns.countplot(x='Coastal',data=data)
# plt.show()

# 然后，观测临海与内陆城市的散点图分布
# sns.swarmplot(x='Coastal',y='AQI',data=data)
# plt.show()

# 最后，分组计算空气质量的均值
# # 方法一：
# aqi_mean=data.groupby(by='Coastal')['AQI'].mean()
# # print(aqi_mean)
# # 方法二：
# sns.barplot(x='Coastal',y='AQI',data=data)
# plt.show()
# 注：
# 柱形图仅能进行均值对比，可以使用箱线图来显示更多信息
# sns.boxplot(x='Coastal',y='AQI',data=data)
# plt.show()
# 还可以绘制小提琴图，功能：除了能展示箱线图的信息外， 还能呈现分布的密度
# sns.violinplot(x='Coastal',y='AQI',data=data)
# plt.show()
# 还可以将散点图与箱线图或小提琴图结合起来金总绘制
# sns.violinplot(x='Coastal',y='AQI',data=data,inner=None)
# sns.swarmplot(x='Coastal',y='AQI',color='r',data=data)
# plt.show()

# sns.boxplot(x='Coastal',y='AQI',data=data)
# sns.swarmplot(x='Coastal',y='AQI',data=data,color='r')
# plt.show()
## 可以得知：临海城市空气质量 优于内陆

'''问题3：空气质量主要受哪些因素的影响'''
# # 相关系数
# corr=data.corr()
# # print(corr.shape)
# # 画图
# plt.figure(figsize=(35,25),dpi=100)
# sns.heatmap(corr,annot=True,fmt='.2f',cmap=plt.cm.RdYlGn)
# plt.xticks(range(1,corr.shape[0]+2),corr.index,fontsize=20,rotation=90)
# plt.yticks(range(corr.shape[1]),corr.index,fontsize=20)
# # plt.savefig('AQI.png',dpi=100)
# plt.show()

##### 可以得知：空气质量指数主要受降雨量（-0.40）与纬度（0.55）影响
# 1)降雨量越多，空气质量越好。
# 2）纬度越低，空气质量越好。
# 此外，还可以得知：
# 1）GDP（城市生产总值）与Incineration（焚烧量）正相关（0.90）。
# 2）Temperature（温度）与Precipitation（降雨量）正相关（0.69）。
# 3）Temperature（温度）与Latitude（纬度）负相关（-0.81）。
# 4）Longitude（经度）与Altitude（海拔）负相关（-0.74）。
# 5）Latitude（纬度）与Precipitation（降雨量）负相关（-0.66）。
# 6）Temperature（温度）与Altitude（海拔）负相关（-0.46）。
# 7）Altitude（海拔）与Precipitation（降雨量）负相关（-0.32）。

'''可疑的相关系数值'''
# 通过前面分析，可知：临海城市的空气质量，确实好于内陆城市。可是（临海 与空气质量指数（AQI）的相关系数（-0.15）并不高）

'''绘制全国城市的空气质量'''
# # 绘制城市的空气质量指数
# # print(data.head(1))
# # 城市位置
# sns.scatterplot(x='Longitude',y='Latitude',hue='AQI',palette=plt.cm.RdYlGn_r,data=data)
# plt.show()
# # 从结果可以发现：从大致的地理位置来看，西部城市要好于 东部城市；南部城市好于北部城市。

'''关于空气质量的假设检验'''
# 假设：全国所有的城市的空气质量指数均在 71 左右，是否可信？？？？
# 问题下的探索：
# 城市平均空气质量指数：
# print(data['AQI'].mean())
# 可以得知 AQI= 75.3343653250774， 但是并不能验证假设，得出正确结论。
# 原因：假设中，指的是全国所有城市的空气质量均值，而此处只是城市中一部分抽样；即一次抽样的均值并不能代表总体的均值。
# 故，要想得知假设是否正确，方法如下：从全国所有城市中进行抽样，使用抽样的均值来估计总体的均值。
'''1.总体与样本的分布'''
# 数学上：
# 如果总体（分布不重要）均值为$\mu$，方差为$\sigma^2$，则样本均值服从正态分布：$\bar{X}$ ~ $N(\mu, \sigma^2 / n)$。
# 其中，n为每次抽样含有个体的数量。
# 我们可以得到如下结论：
# 1. 进行多次抽样（每次抽样包含若干个个体），则每次抽样会得到一个均值，这些均值会围绕在总体均值左右，呈正态分布。
# 2. 样本均值构成正态分布，其均值等于总体均值。
# 3. 样本均值构成正态分布，其标准差等于总体标准差除以$\sqrt{n}$。

# all=np.random.normal(loc=30,scale=50,size=10000)
# # 构造 0 矩阵
# mean_arr=np.zeros(2000)
# for i in range(len(mean_arr)):
#     mean_arr[i]=np.random.choice(all,size=50,replace=False).mean()
# print(mean_arr.mean())
# sns.kdeplot(mean_arr,shade=True)
# # plt.show()

'''2.置信区间'''
# # 根据正态分布的特性，进行概率上的统计
# # 定义标准差
# scale=50
# # 定义数据
# x=np.random.normal(0,scale,size=100000)
# # 定义标准差的倍数，倍数从 1到 3
# for times in range(1,4):
#     y=x[(x>=-times*scale)&(x<=times*scale)]
#     print(len(y)/len(x))

### 可以得知：
# 以均值为中心，在一倍标准差内$(\bar{x} - \sigma, \bar{x} + \sigma)$，包含68%的样本数据。
# 以均值为中心，在二倍标准差内$(\bar{x} - 2\sigma, \bar{x} + 2\sigma)$，包含95%的样本数据。
# 以均值为中心，在三倍标准差内$(\bar{x} - 3\sigma, \bar{x} + 3\sigma)$，包含99.7%的样本数据
### 结论：
# 如果多次抽样，则样本均值构成的正态分布。如果对总体进行一次抽样，则本次抽样个体的均值有95%的概率会在二倍标准差内，仅有5%的概率会在二倍 标准差外。
# 根据小概率事件（很小的概率在一次抽样中基本不会发生）。
# 如果抽样的个体均值落在二倍标准差之外，我们就可以认为，本次抽样来自的总体，该总体 的均值并非是我们所期望的均值。
# 通常：们以二倍标准差作为判定依据，则二倍标准差围成的区间，称为置信区间。该区间，则为接受域，否则为拒绝域。

'''3.假设检验----t检验'''
# 假设检验，其目的是通过收集到的数据，来验证某个假设是否成立。在假设检验中，我们会建立两个完全对立的假设，分别为原假设（零假设）$H_0$与备则假设 （对立假设）$H_1$。
# 然后根据样本信息进行分析判断，得出P值（概率值）。
# 假设检验基于小概率反证法，即认为小概率事件在一次试验中是不会发生的。如果小概率事件发生，则我们就拒绝原假设，而接受备择假设。
# 否则，我们就没有 充分的理由推翻原假设，此时，我们选择去接受原假设。
# t检验，就是假设检验的一种，可以用来检验一次抽样中样本均值与总体均值的比较。
# 其计算方式如下： $t = \frac{\bar{x} - \mu_0}{S_\bar{x}} = \frac{\bar{x} - \mu_0}{S / \sqrt{n}}$
# $\bar{x}$为一次抽样中，所有个体的均值。
# $\mu_0$为待检验的均值。
# $S_\bar{x}$为样本均值的标准差（标准误差）。
# S为一次抽样中，个体的标准差。
# n为一次抽样中，个体的数量。
# t值体现的，就是一次抽样中，个体均值与待检验的总体均值的偏离程度，
# 如果偏离超过一定范围（通产为2倍的标准差），则拒绝原假设，接受备则假设。
# mean=data['AQI'].mean()
# std=data['AQI'].std()
# # print(mean,std)
# t=(mean-71)/(std/np.sqrt(len(data)))
# print(t)
# 可以得知： 偏离均值不足 2倍的标准差，因此，P值应该 大于 5%，无法拒绝原假设。
# 因此，原假设是有一定依据的，
# 此外，也可以通过 scipy提供的相关方法来进行 t 检验的计算， 无需自行计算。
# from scipy import stats
# st=stats.ttest_1samp(data['AQI'],71)
# # print(st)
#
# # 计算 全国所有城市平均空气质量指数的置信区间
# mean=data['AQI'].mean()
# std=data['AQI'].std()
# all_city=mean-1.96*(std/np.sqrt(len(data))),mean+1.96*(std/np.sqrt(len(data)))
# print(all_city)
#
# # 由此，计算出全国所有城市平均空气质量指数， 95%的可能性大致在 70.55---80.12之间
# # 因此，可以将计算的值带入  进行验证。
# print(stats.ttest_1samp(data['AQI'],70.64536585461275))
# print(stats.ttest_1samp(data['AQI'],80.02336479554205))
# ##结果可知：t 值大致为 1.96， P值大致为临界值 5%。

'''问题4：对空气质量指数 进行预测'''
# 对于某些城市， 如果已知降雨量，温度、经纬度等指标，是否能够预测该城市的空气质量指数？？？
###具体做法：
# 1） 根据以往数据，建立一种模式；
# 2）将这种模式应用于 未知的数据，进而进行预测结果。

'''1. 一元线性回归'''
# 回归分析：用来评估变量之间关系的统计过程。用来解释自变量X 与因变量Y的关系。即当自变量X 发生改变时，因变量Y 会如何发生改变。
# 线性回归：是回归分析的一种，评估的自变量X与因变量Y之间是一种线性关系。
# 当只有一个自变量时，称为一元线性回归。当有多个自变量时，称为多元线性回归。
# 例如：以房屋面积（X）与房屋价格（Y）为例，可知，二者是一种线性关系，房屋价格正比于房屋面积，假设比例为：$\hat{y} = w * x$
# 然而，这种线性方程一定是过原点的，即当 X 为0时，y 也一定为0，这可能不符合现实某些场景。
# 为了能够让方程具有更加广泛的适应性，再增加一个 截距，设为 b ,即之前的方程变为：$\hat{y} = w * x + b$   .
# 假设数据集如下：
# 房屋面积 房屋价格
# 30        100
# 40        120
# 40        115
# 50        130
# 50       132
# 60       147
# 线性回归是用来解释自变量与因变量之间的 关系。但是，这种关系并非严格的函数映射关系。
# 从数据集中，可以得知：相同面积的房屋，价格并不完全相同， 但是，也不会相差很大。

'''2.多元线性回归'''
# 例如：影响房屋价格也很可能不只房屋面积一个因素， 可能还有距地铁距离，距市中心距离，房间数量，房屋所在层数，
# 房屋建筑年代等诸多因素，这些因素，对房屋价格影响的力度（权重）是不同的。例如：房屋所在层数对房屋价格的影响就远远不及房屋面积
#因此，使用多个权重来表示多个因素与 房屋价格的关系：
# $\hat{y} = w{1} * x{1} + w{2} * x{2} + w{3} * x{3} + …… + w{n} * x{n} + b$
'''1) 目标'''
# # 首先，从现有的数据集（经验）中，去确定 w 与 b的值。
# # 一旦确定，就能够确定拟合数据的线性方程；
# # 这样，就可以对未知的数据 x(房屋面积，房屋建筑年代等) 进行预测y (房屋价格)。
# # 求解 w 与 b 的依据：找到一组合适的 w 与 b,使得模型得预测值可以与真实值得总体差异最小化。
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# # print(data.shape)
# x=data.drop(['City','AQI'],axis=1)
# y=data['AQI']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# # 预测值
# y_hat=lr.predict(x_test)
# print(lr.score(x_train,y_train))
# print('*'*20)
# print(lr.score(x_test,y_test))
#
# # 画图
# plt.figure(figsize=(35,20),dpi=100)
# plt.plot(y_test.values,'r',label='真实值')
# plt.plot(y_hat,'-g',label='预测值')
# plt.xticks(range(0,90,10),fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.title('线性回归预测结果',fontsize=25)
# plt.show()

'''2) 对是否临海进行 预测'''
# 对于某些城市，假设是否临海未知， 但知道其他信息，我们试图使用其他信息， 来预测该城市是否 临海。
'''逻辑回归'''
# 逻辑回归，实际上，逻辑回归是一个分类算法。
# 其优点在于，逻辑回归不仅能够进行分类，而且还能够获取属于该类别的概率。
# 这在现实 中是非常实用的。例如，某人患病的概率，明天下雨的概率等。
# 逻辑回归实现分类的思想为：将每条样本进行“打分”，然后设置一个阈值，达到这个阈值的，分为一个类别，而没有达到这个阈值的，分为另外一个类别。
# 对于阈值， 比较随意，划分为哪个类别都可以，但是，要保证阈值划分的一致性。

'''算法模拟'''
# 对于逻辑回归，模型的前面与线性回归类似：$z = w_1x_1 + w_2x_2 + …… + w_nx_n + b$
# 不过，z的值是一个连续的值，取值范围为$(-\infty , +\infty)$我们需要将其转换为概率值，逻辑回归使用sigmoid函数来实现转换，
# 该函数的原型为： $sigmoid(z) = \frac{1}{1 + e^{-z}}$
# 当z的值从$-\infty$向$+\infty$过度时，sigmoid函数的取值范围为[0, 1]，这正好是概率的取值范围，
# 当$z=0$时，sigmoid(0)的值为0.5。因此，模型就可以将sigmoid的输出p作为正例的概率，而1 - p作为负例的概率。
# 以阈值0.5作为两个分类的标准，假设真实的分类y 的值为1与0，则： $ \hat y = \left{\begin{matrix} 1\quad p >= 0.5\ 0\quad p < 0.5 \end{matrix}\right. $
# 因为概率p就是sigmoid函数的输出值，因此有： $ \hat y = \left{\begin{matrix} 1\quad sigmoid(z) >= 0.5\ 0\quad sigmoid(z) < 0.5 \end{matrix}\right. $
# 也可以表示为： $ \hat y = \left{\begin{matrix} 1\quad z >= 0\ 0\quad z < 0 \end{matrix}\right. $
# print(data.info())
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x=data.drop(['City','Coastal'],axis=1)
y=data['Coastal']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
lr=LogisticRegression(C=0.0001)
lr.fit(x_train,y_train)
y_hat=lr.predict(x_test)
# print(lr.score(x_train,y_train))
# print(lr.score(x_test,y_test))

# #画图
# plt.figure(figsize=(35,20),dpi=100)
# plt.plot(y_test.values,marker='o',c='r',ms=8,ls='',label='真实值')
# plt.plot(y_hat,marker='x',color='g',ms=8,ls='',label='预测值')
# plt.legend(fontsize=25)
# plt.title('逻辑回归预测结果',fontsize=25)
# # plt.show()


probability=lr.predict_proba(x_test)
# print(probability[:10])
# print(np.argmax(probability,axis=1))
index=np.arange(len(x_test))
pro_0=probability[:,0]
pro_1=probability[:,1]
tick_label=np.where(y_test==y_hat,'0','x')
# 画图 （绘制堆叠图）
plt.figure(figsize=(35,20),dpi=100)
plt.bar(index,height=pro_0,color='g',label='类别0概率值')
# bottom=x, 表示从 x 的值开始堆叠上去
# tick_label 设置标签刻度的文本内容
plt.bar(index,height=pro_1,color='r',bottom=pro_0,label='类别1概率值',tick_label=tick_label)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('样本序号',fontsize=25)
plt.ylabel('各个类别的概率',fontsize=25)
plt.title('逻辑回归分类概率',fontsize=25)
plt.legend(loc='best',fontsize=25)
plt.show()

'''结论'''
# 1. 空气质量总体分布上来说，南部城市优于北部城市，西部城市优于东部城市。
# 2. 临海城市的空气质量整体上好于内陆城市。
# 3. 是否临海，降雨量与纬度对空气质量指数的影响较大。
# 4. 我国城市平均空气质量指数大致在(70.55 ~ 80.12)这个区间内，在该区间的可能性概率为95%。
# 5. 通过历史数据，我们可以对空气质量指数进行预测。
# 6. 通过历史数据，我们可以对城市是否临海进行预测。







