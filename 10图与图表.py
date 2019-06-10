'''
创建图与图表是很多分析项目中的一个重要步骤，它通常是项目开始时探索性数据分析 （EDA）的一部分，
或者在项目报告阶段向其他人介绍你的数据分析结果时使用。数据可 视化可以使你看到变量的分布和变量之间的关系，还可以检查建模过程中的假设
'''
'''
一、matplotlib
# 功能：1）matplotlib 是一个绘图库，创建标准统计图，创建的图形可达到出版的质量要求。它可以创建常用的统计 图，包括条形图、箱线图、折线图、散点图和直方图。
# 它还有一些扩展工具箱，比如 basemap 和 cartopy，用于制作地图，以及 mplot3d，用于进行 3D 绘图。
# 2）matplotlib 提供了对图形各个部分进行定制的功能。例如，它可以设置图形的形状和大 小、x 轴与 y 轴的范围和标度、x 轴与 y 轴的刻度线和标签、图例以及图形的标题。
# '''
# # 1. 条形图 （计数值）
# # 分类：条形图包括垂直图、水平图、堆积图和分 组图
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# customers=['ABC','DEF','GHI','JKL','MNO']
# customers_index=[i for i in range(len(customers))]
# sale_amounts=[127,90,201,111,232]
# # 创建基础图
# fig=plt.figure()
# # 向基础图中添加子图
# ax1=fig.add_subplot(1,1,1)    # 1, 1, 1 表示创建 1 行 1 列的子图，并使用第 1 个也是唯一的一个子图
# ax1.bar(customers_index,sale_amounts,align='center',color='darkblue')
# ax1.xaxis.set_ticks_position('bottom')     # 设置刻度线 x 轴位置
# ax1.yaxis.set_ticks_position('left')       # 设置刻度线 y 轴 位置
# # 将条形的刻度线标签 由客户索引值改为 实际的客户名称 ；
# # rotation=0 表示刻度标签是水平的
# plt.xticks(customers_index,customers,rotation=0,fontsize='small')
# plt.xlabel('Customer Name')
# plt.ylabel('Sale')
# plt.title('Sale Amount per Customer')
# # dpi=400 : 设置图形分辨率；
# # bbox_inches='tight' ：表示在保存图形时，将图形四周的空白部分去掉
# plt.savefig('bar_plot.png',dpi=500,bbox_inches='tight')
# plt.show()


# # 2.直方图（数值分布）
# # 分类：常用的直方图包括频率分布、频率密度分布、概率分布和概率 密度分布
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# mu1,mu2,sigma=100,130,15
# x1=mu1+sigma*np.random.randn(10000)
# x2=mu2+sigma*np.random.randn(10000)
# # 基础图
# fig=plt.figure()
# # 子图
# ax1=fig.add_subplot(1,1,1)
# n,bins,patches=ax1.hist(x1,bins=50,density=False,color='darkgreen')
# # alpha=0.5 :是透明的  ; density=False :表示直方图显示 频率分布，而不是概率密度 ；  bins=50 ：表示变量分成50分
# n1,bins1,patches1=ax1.hist(x2,bins=50,density=False,color='orange',alpha=0.5)
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# plt.xlabel('Bins')
# plt.ylabel('Number of Value in Bin')
# fig.suptitle('Histograms',fontsize=14,fontweight='bold')
# ax1.set_title('\nTwo Frequency Distributions')
# plt.show()

# # 3.折线图（反应数据随时间的变化）
# import matplotlib.pyplot as plt
# from numpy.random import randn
# plt.style.use('ggplot')
# plot_data1=randn(50).cumsum()
# plot_data2=randn(50).cumsum()
# plot_data3=randn(50).cumsum()
# plot_data4=randn(50).cumsum()
# # 基础图
# fig=plt.figure()
# #子图
# ax1=fig.add_subplot(1,1,1)
# ax1.plot(plot_data1,marker='o',color='blue',linestyle='-',label='Blue Solid')
# # plt.show()
# ax1.plot(plot_data2,'+',color='red',linestyle='--',label='Red Dashed')
# ax1.plot(plot_data3,'*',color='green',linestyle='-.',label='Green Dash Dot')
# ax1.plot(plot_data4,'s',color='orange',linestyle=':',label='Orange Dotted')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# # 数据点类型、 颜色、 线型
# ax1.set_title('Line Plots:Markers,Color,and Linestyles')
# plt.xlabel('Draw')
# plt.ylabel('Random Number')
# plt.legend(loc='best')
# plt.show()

# # 4.散点图
# # 作用：散点图表示两个数值变量之间的相对关系，这两个变量分别位于两个数轴上。
# # 例如，身高 与体重，或者供给与需求。散点图有助于识别出变量之间是否具有正相关（图中的点集中 于某个具体参数）或负相关（图中的点像云一样发散）。
# # 你还可以画一条回归曲线，也就 是使方差最小的曲线，通过图中的点基于一个变量的值预测另一个变量的值。
#
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# x=np.arange(start=1,stop=15,step=1)
# # 通过随机数使数据与一条直线和一条二 次曲线稍稍偏离
# y_liner=x+5*np.random.randn(14)             # 直线
# y_quadratic=x**2+10.*np.random.randn(14)    # 二次曲线
# # 数据拟合
# fn_liner=np.poly1d(np.polyfit(x,y_liner,deg=1))
# fn_quadratic=np.poly1d(np.polyfit(x,y_quadratic,deg=2))
# # 基础图
# fig=plt.figure()
# #子图
# ax1=fig.add_subplot(1,1,1)
# ax1.plot(x,y_liner,'bo',x,y_quadratic,'go',x,fn_liner(x),'b-',x,fn_quadratic(x),'g-',linewidth=2.)
# #设置位置
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# ax1.set_title('Scatter Plots Regression Lines')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.xlim(min(x)-1,max(x)+1)
# plt.ylim(min(y_quadratic)-10,max(y_quadratic)+10)
# plt.show()

# # 5.箱线图
# #箱线图可以表示出数据的最小值、第一四分位数、中位数、第三四分位数和最大值。
# # 箱体 的下部和上部边缘线分别表示第一四分位数和第三四分位数，箱体中间的直线表示中位 数。
# # 箱体上下两端延伸出去的直线（whisker，亦称为“须”）表示非离群点的最小值和最 大值，在直线（须）之外的点表示离群点。
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# N=500
# normal=np.random.normal(loc=0.0,scale=1.0,size=N)
# lognormal=np.random.lognormal(mean=0.0,sigma=1.0,size=N)
# index_value=np.random.randint(low=0,high=N-1,size=N)
# normal_sample=normal[index_value]
# lognormal_sample=lognormal[index_value]
# box_plot_data=[normal,normal_sample,lognormal,lognormal_sample]
# fig=plt.figure()
#
# ax1=fig.add_subplot(1,1,1)
# # 创建箱线图 标签
# box_labels=['normal','normal_sample','lognormal','lognormal_sample']
# # notch=False 表示箱体是 矩形， 而不是中间缩放；  sym='.' 表示离群点使用圆点； 而不是默认的“+”
# # vert=True 表示箱体是垂直的；   showmeans=True 表示既显示中位数 又显示均值
# ax1.boxplot(box_plot_data,notch=False,sym='.',vert=True,whis=1.5,showmeans=True,labels=box_labels)
#
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# ax1.set_title('Box Plots:Resampleing of Two Distributions')
# ax1.set_xlabel('Distribution')
# ax1.set_ylabel('Value')
# plt.show()

'''
二、pandas 
作用：pandas 通过提供一个可以作用于 序列和数据框的函数plot， 简化了基于序列和数据框中的数据创建图表的过程；
plot 函数默认创建折线图，可通过设置参数 kind 创建 其他类型的图表
用途：除了使用 matplotlib 创建标准统计图，还可以使用 pandas 创建其他类型的统计图， 
比如六边箱图（hexagonal bin plot）、矩阵散点图、密度图、Andrews 曲线图、平行坐标图、 延迟图、自相关图和自助抽样图。
如果要向统计图中添加第二y 轴、误差棒和数据表，使 用 pandas 可以很直接地实现。
'''
# # 1.创建一对条形图和箱线图，并将他们 并排放置
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# # 创建基础图 和 两个 子图
# fig,axes=plt.subplots(1,2)
# # 将两个子图赋予变量 ax1,ax2
# ax1,ax2=axes.ravel()
# data_frame=pd.DataFrame(np.random.rand(5,3),index=['Customer 1','Customer 2','Customer 3','Customer 4','Customer 5'],
#                         columns=pd.Index(['Metric 1','Metric 2','Metric 3'],name='Metrics'))
#
# # 使用 pandas 的 plot 函数在左侧子图中创建一个条形图
# data_frame.plot(kind='bar',ax=ax1,alpha=0.75,title='Bar Plot')
# # 使用 matplotlib 的函数来设置 x 轴和 y 轴标签的旋转角度和字体大小。
# plt.setp(ax1.get_xticklabels(),rotation=45,fontsize=14)
# plt.setp(ax1.get_yticklabels(),rotation=0,fontsize=14)
# ax1.set_xlabel('Customer')
# ax1.set_ylabel('Value')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
#
# # 为箱线图单独创建一个 颜色字典
# colors=dict(boxes='DarkBlue',whiskers='Gray',medians='Red',caps='Black')
# # 在右侧子图中创建箱线图，使用 colors 为箱线图各部分着色，并将离群点的形状设置为红色圆点
# data_frame.plot(kind='box',color=colors,sym='r.',ax=ax2,title='Box Plot')
# plt.setp(ax2.get_xticklabels(),rotation=45,fontsize=14)
# plt.setp(ax2.get_yticklabels(),rotation=0,fontsize=14)
# ax2.set_xlabel('Metric')
# ax2.set_ylabel('Value')
# ax2.xaxis.set_ticks_position('bottom')
# ax2.yaxis.set_ticks_position('left')
#
# plt.show()

'''
三、ggplot （不常用）
解释：ggplot 是一个Python 绘图包，它基于R 语言的 ggplot2 包和图形语法。ggplot 与其他绘 图包的关键区别是它的语法将数据与实际绘图明确地分离开来。
。为了对数据进行可视化表 示，ggplot 提供了几种基本元素：几何对象、图形属性和标度。除此之外，为了进行更高 级的绘图，ggplot 还提供一些附加元素：统计变换、坐标系、子窗口和可视化主题。
'''
#通过 ggplot 创建一些基础统计图，使用的数据就是 ggplot 包内部的数据





'''
四、seaborn
解释：seaborn 简化了在 Python 中创建信息丰富的统计图表的过程。
它是在 matplotlib 基础上开 发的，支持 numpy 和 pandas 中的数据结构，并集成了 scipy 和 statsmodels 中的统计程序
功能：seaborn 可以创建标准统计图，包括直方图、密度图、条形图、箱线图和散点图。它可 以对成对变量之间的相关性、线性与非线性回归模型以及统计估计的不确定性进行可视 化。
它可以用来在评估变量时检查变量之间的关系，并可以建立统计图矩阵来显示复杂 的关系。它有内置的主题和调色板，可以用来制作精美的图表。
最后，因为它是建立在 matplotlib 上的，所以你可以使用 matplotlib 的命令来对图形进行更深入的定制。
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig
sns.set(color_codes=True)
# #1.创建  直方图
# x=np.random.normal(size=100)  # 生成100 个随机数
# sns.distplot(x,bins=20,kde=False,rug=True,label='Histogram w/o Density')
# plt.title("Histogram of a Random Sample from a Normal Distribution")
# plt.legend()
# # plt.show()

# #2.创建 带有回归直线的散点图与单变量直方图
# mean,cov=[5,10],[(1,.5),(.5,1)]
# data=np.random.multivariate_normal(mean,cov,200)
# data_frame=pd.DataFrame(data,columns=['x','y'])
# sns.jointplot(x='x',y='y',data=data_frame,kind='reg').set_axis_labels('x','y')
# plt.suptitle('"Joint Plot of Two Variables with Bivariate and Univariate Graphs')
# plt.show()

# #3.成对变量之间的散点图与 单变量直方图
# iris=sns.load_dataset('iris')
# sns.pairplot(iris)
# plt.show()

# # 4.按照 某 几个变量生成的箱线图
# tips=sns.load_dataset('tips')
# sns.factorplot(x='time',y='total_bill',hue='smoker',col='day',data=tips,kind='box',size=4,aspect=.5)
# # plt.show()

# # 5.带有 bootstrap 置信区间的线性回归模型
# sns.lmplot(x='total_bill',y='tip',data=tips)
# # plt.show()
# # 带有 bootstrap 置信区间的逻辑斯蒂回归模型
# tips['big_tip']=(tips.tip/tips.total_bill)>.15
# sns.lmplot(x="total_bill", y="big_tip", data=tips, logistic=True, y_jitter=.03).set_axis_labels("Total Bill", "Big Tip")
# plt.title("Logistic Regression of Big Tip vs. Total Bill")
# plt.show()























