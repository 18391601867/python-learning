'''
描述性统计与建模
目标：1）使用统计图和 摘要统计量对数据集 进行探索和摘要分析；
2） 使用多元线性回归和 逻辑斯蒂回归 进行 回归和分类分析
3）使用 pandas 和 statsmodels生成 标准的描述性统计量 和模型
'''
'''
项目一：以葡萄酒数据为例：
具体步骤:
'''
'''1.创建数据集'''
# 注：创建具有成千上万行数据的数据集，不需从零开始，从互联网上下载即可。
# 我们要使用的 第一个数据集是葡萄酒质量数据集，从 UCI 机器学习资料库中可以找到。
# 第二个数据集是 客户流失数据集，来自于几个数据分析博客。

# 第一个数据集：
# 葡萄酒质量数据集 详见 winequality-both.csv 红葡萄酒与白葡萄酒数据的组合；
# 可以发现该数据：有 1 个输出变量和 11 个输 入变量。输出变量是酒的质量，是一个从 0（低质量）到 10（高质量）的评分。
# 输入变量 是葡萄酒的物理化学成分和特性，包括非挥发性酸、挥发性酸、柠檬酸、残余糖分、氯化 物、游离二氧化硫、总二氧化硫、密度、pH 值、硫酸盐和酒精含量。

# 第二个数据集：
#客户流失数据集  详见 churn.csv
# 可以发现该数据：件有1 个输出变量和20 个输入变量。
# 输出变量 Churn? 是一个布尔型变量 （True/False），表示在数据收集的时候，客户是否已经流失（是否还是电信公司的客户）。
# 输入变量是客户的电话计划和通话行为的特征，包括状态、账户时间、区号、电话号码、 是否有国际通话计划、是否有语音信箱、
# 语音信箱消息数量、白天通话时长、白天通话 次数、白天通话费用、傍晚通话时长、傍晚通话次数、傍晚通话费用、夜间通话时长、
# 夜间通话次数、夜间通话费用、国际通话时长、国际通话次数、国际通话费用和客户服 务通话次数。

'''2.数据探索分析'''
# # 2.1 描述性统计
# # 第一步：分析葡萄酒质量总数据集：
# # 目标：计算出每列的总体描述性统计量、 质量列中的唯一值和 这个唯一值对应的观测数据
# # 程序代码如下：
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from statsmodels.formula.api import ols,glm
# # 第一步：读取数据到 pandas 数据框
# input_file='winequality-both.csv '
# wine=pd.read_csv(input_file,sep=',',header=0)     # sep=',' 表示域分隔符为逗号， header=0 表示第一行为列标题
# wine.columns=wine.columns.str.replace(' ','_')    # 如果列标题中有空格，则使用 下划线 替代空格
# # print(wine.head())       # 使用head函数检查一下标题行和 前 5行数据 ：确保数据被正确加载
# # 第二步：显示所有变量的描述性统计量
# #些统计量包括总数、均值、标准差、最小值、第 25 个百分位数、中位数、第 75 个百分 位数和最大值。
# # 例如，质量评分中有 6497 个观测，评分范围从 3 到 9，平均质量评分为 5.8，标准差为 0.87。
# # print(wine.describe())
# # 第三步：找出唯一值
# # print(sorted(wine.quality.unique()))   # 找出质量列 中的唯一值
# # 第四步：计算值得频率
# # print(wine.quality.value_counts())      # 找出 相同的值出现的 次数
#
# # 2.2 分组、直方图、与 t 检验
# #目标：分别分析红葡萄酒和白葡萄酒数据 ，观察统计量是否 会保持不变
# # 第一步：按照葡萄酒类别显示 质量的 描述性统计量
# # groupby函数使用 'type'，将数据分为 红酒和白酒两组； unstack函数将结果重新排列，目的使红、白酒 并排在两列
# # print(wine.groupby('type')[['quality']].describe().unstack('type'))
# # 第二步：按照葡萄酒的类别显示质量的 特定分位数值
# # print(wine.groupby('type')[['quality']].quantile([0.25,0.75]).unstack('type'))
# # 第三步：按照葡萄酒类别查看 质量分布
# red_wine=wine.loc[wine['type']=='red','quality']
# # print(red_wine)   #红酒质量
# white_wine=wine.loc[wine['type']=='white','quality']
# # print(white_wine)  # 白酒质量
#
# # # 第四：使用 seaborn 创建统计图
# # # sns.set(color_codes=True)
# # sns.set_style('dark')
# # print(sns.distplot(red_wine,norm_hist=True,kde=False,color='red',label='Red Wine'))
# # print(sns.distplot(white_wine,norm_hist=True,kde=False,color='white',label='White Wine'))
# # plt.ylabel('Density')
# # plt.title("Distribution of Quality by Wine Type")
# # plt.legend(loc='best')
# # plt.show()
# # 结论分析：红条表示红葡萄酒，白条表示白葡萄酒。
# # 因为白葡萄酒数据比红葡萄酒多（白葡萄酒有 4898 条数据，红葡萄酒有 1599 条数据），所以图中显示密度分布，不显示频率分布。
# # 从这 个统计图可以看出，两种葡萄酒的评分都近似正态分布。
# # 与原始数据的摘要统计量相比， 直方图更容易看出两种葡萄酒的质量评分的分布。
#
#
# #第五步： 检验红、白酒的平均质量是否有所不同 ( t 检验)
# # t 检验方法：使用合并方差
# # t 检验统计量为 -9.69，p 值为 0.00，这说明白葡萄酒的平均质量评分在统计意义上大于红 葡萄酒的平均质量评分。
# # print(wine.groupby('type')[['quality']].agg(['std']))
# # tstat,pvalue,df=sm.stats.ttest_ind(red_wine,white_wine)
# # print('tstat:%0.2f pvalue:%.3f' % (tstat,pvalue))
#
# #2.3 成对变量之间的关系和 相关性
# # 目标：计算输入变量 两两之间的相关性 ， 并 为一些输入变量创建带有回归 直线的散点图
#
# # 第一步：计算所有变量的 相关矩阵
# # print(wine.corr())
# # 第二步：从红、白 葡萄酒的数据中 取出 一个“小”样本数据来进行绘图
# def take_sample(data_frame,replace=False,n=200):
#     return data_frame.loc[np.random.choice(data_frame.index,replace=replace,size=n)]
# # 对 红、白 葡萄酒进行随机抽样
# reds_sample=take_sample(wine.loc[wine['type']=='red',:])
# # print(reds_sample)
# whites_sample=take_sample(wine.loc[wine['type']=='white',:])
# # print(whites_sample)
# wine_sample=pd.concat([reds_sample,whites_sample])  # 将抽样所得的两个数据框 连接成一个数据框
# # 在 wine 数据框创建一个新列 ，使用 numpy 中的where 和 pandas 中的 isin 函数对这个新列进行填充；
# # 填充的值根据此行的索引值是否在抽 样数据的索引值中分别设为1 和 0
# wine['in_sample']=np.where(wine.index.isin(wine_sample.index),1,0)
# # print(pd.crosstab(wine.in_sample,wine.type,margins=True))
#
# # 查看成对变量之间的 关系
# sns.set_style('dark')
# # pairplot 函数创建一个统计图矩阵。主对角线上的图以 直方图或 密度图的 形势显示了每个变量的单变量分布；
# # 对角线之外的图以散点图 的形式显示了 每两个变量之间的双变量分布，散点图中可以有回归直线，也可以没有
# g=sns.pairplot(wine_sample,kind='reg',plot_kws={'ci':False,'x_jitter':0.25,'y_jitter':0.25},hue='type',diag_kind='hist',\
#                diag_kws={'bins':10,'alpha':1.0},palette=dict(red='red',white='white'),markers=['o','s'],vars=['quality','alcohol','residual_sugar'])
# # print(g)
# plt.suptitle('Histograms and Scatter Plots of Quality,  Alcohol, and Residual\ Sugar', fontsize=14, horizontalalignment='center', verticalalignment='top',x=0.5, y=0.999)
# # plt.show()
#
# # 注：统计图显示了葡萄酒质量、酒精含量和残余糖分之间的关系。红条和红点表示 红葡萄酒，白条和白点表示白葡萄酒。
# # 因为质量评分都是整数，所以加上了一点振动， 这样更容易看出数据在何处集中。
#
# # 结论分析
# # 从这些统计图可以看出，对于红葡萄酒和白葡萄酒来说，酒精含量的均值和标准差是大致 相同的，但是，白葡萄酒残余糖分的均值和标准差却大于红葡萄酒残余糖分的均值和标准 差。
# # 从回归直线可以看出，对于两种类型的葡萄酒，酒精含量增加时，质量评分也随之提 高，相反，残余糖分增加时，质量评分则随之降低。
# # 这两个变量对白葡萄酒的影响都要大 于对红葡萄酒的影响。
#
# # 2.4 使用最小二乘法估计 进行线性回归
# # 相关关系和 两两变量之间的统计图 有助于对两个变量之间的关系进行量化和 可视化
# # 但是，他们 不能测量出 每个自变量在其他自变量不变时 与因变量之间的关系。。
# # 为了解决这个问题：利用线性回归
# #解释：
# #  yi ~ N(μi,σ2),
# #  μi = β0 + β1xi1 + β2xi2 +… + βpxip
# #  对于 i = 1, 2, …, n 个观测和 p 个自变量。
# #  这个模型表示观测 yi 服从均值为 μi 方差为 σ2 的正态分布（高斯分布），其中 μi 依赖于自变 量，σ2 为一个常数。
# #  也就是说，给定了自变量的值之后，我们就可以得到一个具体的质量 评分，但在另一天，给定同样的自变量值，我们可能会得到一个和前面不同的质量评分。
# #  但是，经过很多天自变量取同样的值（也就是一个很长的周期），质量评分会落在 μi±σ 这 个范围内。
#
# # 程序如下：
# # 注：。波浪线（~）左侧的变量 quality 是因变量，波浪线右侧的变量是自变量
# my_formula='quality ~alcohol + chlorides + citric_acid + density + fixed_acidity + free_sulfur_dioxide + pH + ' \
#            'residual_sugar + sulphates +total_sulfur_dioxide + volatile_acidity'
# # # 使用公式和数据拟合一个普通最小二乘回归模型
# # lm=ols(my_formula,data=wine).fit()
# # 也 可以使用广义线性模型（glm）的语法 代替普通最小二乘语法，拟合同样的模型。
# lm=glm(my_formula,data=wine,family=sm.families.Gaussian()).fit()
#
# '''打印模型结果'''
# # 打印 摘要信息 :包含了模型系数、系数的标准差和置信区间、修正 R 方、F 统计量等 模型详细信息
# # print(lm.summary())
#
# # 打印出列表 :
# # 包含从模型对象 lm 中提取出的所有数值信息，检查了 这个列表之后，提取出模型系数、系数的标准差、修正R 方、F 统计量和它的p 值，以及模型拟合值。
# # print('所有数值信息：%s' % dir(lm))
#
# # 以一个序列的形式 返回模型系数
# # print(lm.params)
# # 以序列的形式 返回模型系数 的标准差
# # print(lm.bse)
# # lm.fittedvalues 返回拟合值
# # print(lm.fittedvalues)
#
# # 2.5 系数解释
# # 如果你想使用这个模型弄清楚因变量（葡萄酒质量）和自变量（11 个葡萄酒特性）之间的 关系，就应该解释一下模型系数的意义。
# # 在这个模型中，某个自变量系数的意义是，在其 他自变量保持不变的情况下，这个自变量发生 1 个单位的变化时，导致葡萄酒质量评分发 生的平均变化。
# # 例如，酒精含量系数的含义就是，从平均意义上来说，如果两种葡萄酒其 他自变量的值都相同，
# # 那么酒精含量高 1 个单位的葡萄酒的质量评分就要比另一种葡萄酒 的质量评分高出 0.27 分。
#
# # 2.6 自变量标准化
# # 关于这个模型，还需要注意的一点是，普通最小二乘回归是通过使残差平方和最小化来 估计未知的 β 参数值的，
# # 这里的残差是指自变量观测值与拟合值之间的差别。因为残差 大小是依赖于自变量的测量单位的，
# # 所以如果自变量的测量单位相差很大的话，那么将 自变量标准化后，就可以更容易对模型进行解释了。
# # 对自变量进行标准化的方法是，先 从自变量的每个观测值中减去均值，然后再除以这个自变量的标准差。
# # 自变量标准化完成以后，它的均值为 0，标准差为 11。
#
# # print(wine.describe())
# # 使用 wine.describe()，可以看到氯化物的范围是从 0.009 到 0.661，而总二氧化硫的范围 是从 6.0 到 440.0。其余各变量的最小值与最大值之间的区别也大致如此。
# # 因为各个自变量 值的变化范围相差非常悬殊，所以非常应该对自变量进行标准化，看看这样做了之后，能 否更容易对结果进行解释。
#
# # 程序如下：
# # 首先，创建一个序列 来保存质量数据
# dependent_var=wine['quality']
# # 其次， 创建一个数据框 来保存初始的葡萄酒数据集中 除了 quality、 type、 in_sample 之外的变量
# independent_var=wine[wine.columns.difference(['quality','type','in_sample'])]
#
# # 然后，对自变量进行标准化
# # 方法：对每个变量，在每个观测值中 减去 变量的均值； 然后 使用结果 除以 变量的标准差
# independent_var_standardizd=(independent_var-independent_var.mean())/independent_var.std()
# # print(independent_var_standardizd)
# # 最后，将因变量 quality 作为 一列 添加到 自变量数据框中
# # 创建一个 带有标准化自变量的 新数据集
# wine_standerdized=pd.concat([dependent_var,independent_var_standardizd],axis=1)
# # print(wine_standerdized)
#
# # 最后，完成带有标准化自变量的数据集后， 重新进行线性回归， 并查看 摘要统计
# lm_standerdized=ols(my_formula,data=wine_standerdized).fit()
# # print(lm_standerdized.summary())
#
# # 解释：自变量标准化会改变我们对模型系数的解释。现在每个自变量系数的含义是，
# # 不同的葡萄 酒在其他自变量均相同的情况下，某个自变量相差 1 个标准差，会使葡萄酒的质量评分平 均相差多少个标准差。
# # 举个例子，酒精含量系数的意义是，从平均意义上说，如果两种葡 萄酒其他自变量的值都相同，
# # 那么酒精含量高 1 个标准差的葡萄酒的质量评分就要比另一 种葡萄酒的质量评分高出 0.32 个标准差。
#
# # 还是通过 wine.describe() 函数，我们可以看到酒精含量的均值和标准差是 10.5 和 1.2，质
# # 量评分的均值和标准差是 5.8 和 0.9。因此，
# # 从平均意义上说，如果两种葡萄酒其他的自变 量值均相同，那么酒精含量为 11.7（10.5+1.2）的葡萄酒的质量评分就会比酒精含量为均 值的葡萄酒的质量评分大 0.32 个标准差。
#
# # 2.7 预测
# # 在某些情况下，我们需要使用没有用来拟合模型的新数据进行预测。例如，你会收到关于 葡萄酒成分的一个新的观测，
# # 并需要根据这些成分预测这种葡萄酒的质量评分。让我们通 过选择现有数据集的前 10 个观测，并根据它们的葡萄酒成分预测质量评分，
# # 来演示一下 如何对新数据做出预测。
# # 程序如下：
# # 使用葡萄酒数据集中的 前 10个观测值创建 10个 “新”观测
# # 新观测 中只包含模型中使用的 自变量
# new_observations=wine.loc[wine.index.isin(range(10)),independent_var.columns]
# # 基于新观测中的 葡萄酒特性预测质量评分
# y_predicted=lm.predict(new_observations)
#
# # print(round(y_predicted,2))

''' 3.客户流失 '''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
'''读入数据'''
# 读入数据到数据框
churn=pd.read_csv('churn.csv',sep=',',header=0)
# print(churn['State'])
# 格式化列标题
# 两次使用 replace 函 数将列标题中的空格替换成下划线，并删除嵌入的单引号;
# 使用 strip 函数除去了列标题 Churn? 末尾的问号;
# 使用列表生成式将所有列标题转换为小写。
churn.columns=[heading.lower() for heading in churn.columns.str.replace(' ','_').str.replace("\'","").str.strip('?')]
# print(churn.columns)
# 使用 numpy 的 where函数 根据churn 这一列中的值 用 1 或 0 来填充；
# churn01 中的值就是 1，如果 churn 中的值是 False，那么 churn01 中的值就是 0。
churn['churn01']=np.where(churn['churn']=='True.',1,0)
# print(churn.head(20))

'''分组：计算流失客户和未流失客户的描述性统计量； '''
# 对总数、均值、标准差：分析区别
# print(churn.groupby(['churn'])[['day_charge','eve_charge','night_charge','intl_charge','account_length','custserv_calls']].agg(['count','mean','std']))
# 对结果进行并排 unstack(['churn'])
# print(churn.groupby(['churn'])[['day_charge','eve_charge','night_charge','intl_charge','account_length','custserv_calls']].agg(['count','mean','std']).unstack(['churn']))

# 为不同的变量 计算不同的统计量
# print(churn.groupby(['churn']).agg({'day_charge':['mean','std'],'eve_charge':['mean','std'],'night_charge':['count','mean','std'],
#     'intl_charge':['mean','std'],'account_length':['count','min','max'],'custserv_calls':['count','min','max']}).unstack(['churn']))

#
# 创建 total_charges
# 表示白天、傍晚、夜间和国际通话费用的总和
churn['total_charges']=churn['day_charge']+churn['eve_charge']+churn['night_charge']+churn['intl_charge']
# print(churn['total_charges'])

'''等宽分箱法'''
#对客户服务通话次数 这部分数据 进行了 摘要分析
# 将其分为 5组，并为 每一组计算统计量
#  方法：使用 cut 函数 按照等宽分箱法将 total_charges 分成 5 组。
factor_cut=pd.cut(churn.total_charges,5,precision=2)   # precision=2 精确度
# print(factor_cut)
# 定义一个 函数 get_stats, 为每个分组返回 一个 统计量 字典
def get_stats(group):
    return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean(),'std':group.std()}
# 按照 5个 total_charges 分组将 客户服务通话次数也 分成同样的5 组；
# 在分组数据上 应用 get_stats 函数，为 5个 分组 计算统计量
# grouped=churn.custserv_calls.groupby(factor_cut)
# print(grouped.apply(get_stats).unstack())

'''等深分箱法'''
# 用 5个统计量对客户服务通话数据 进行了摘要分析
# 方法： 使用 qcut函数 通过等深分箱法（按照分位数进行划分）将 account_length分成了4组
# 将 account_length 按照分位数 进行分组
# factor_qcut=pd.qcut(churn.account_length,[0,0.25,0.5,0.75,1])
# # 并为每个分位数分组计算统计量
# grouped=churn.custserv_calls.groupby(factor_qcut)
# print(grouped.apply(get_stats).unstack())
'''等宽分箱法 和 等深分箱法的区别'''
# 通过分位数对 account_length 进行划分，可以保证每个分组中包含数目大致相同的观测。
# 通过等宽分箱法得到的每个分组中包含的观测数目是不一样的；
# qcut 函数使 用一个整数或一个分位数数组来设定分位数的数量，所以你可以使用整数 4 来代替 [0., 0.25, 0.5, 0.75, 1.] 设定 4 等分，
# 或使用 10 来设定 10 等分。

'''使用 get_dummies函数创建 二值指标变量'''
# # 为 intl_plan 和 vmail_plan 创建 二值（虚拟）指标变量
# intl_dummies=pd.get_dummies(churn['intl_plan'],prefix='intl_plan')
# vmail_dummies=pd.get_dummies(churn['vmail_plan'],prefix='vmail_plan')
# # 将他们与 新数据框中的 churn 列连接起来
# churn_with_dummies=churn[['churn']].join([intl_dummies,vmail_dummies])
# print(churn_with_dummies.head())

'''如何将一列 按照四分位数进行划分，并为每个四分位数创建二值指标变量，并将新列添加到原来的数据框中'''
# # 将 total_charges 按照四分位数 分组，
# qcut_names=['1st_quartile','2nd_quartile','3rd_quartile','4th_quartile']
# total_charges_quartiles=pd.qcut(churn.total_charges,4,labels=qcut_names)
# # print(total_charges_quartiles)
# # 并为每个 分位数分组创建一个 二值指标变量, 将  total_charges 作为新列的前缀；
# # 最后结果为：4个新的 虚拟变量
# dummies=pd.get_dummies(total_charges_quartiles,prefix='total_charges')
# # join 函数将 4个变量 追加到数据框 churn 中
# churn_with_dummies=churn.join(dummies)
# print(churn_with_dummies.head())

'''创建透视表'''
# 对 total_charges 列按照流失情况和 客户服务通话次数进行透视转换， 计算你每组的 均值；
# 结果表示 每个流失情况 和 客户服务通话次数组合情况的 平均总费用；
# print(churn.pivot_table(['total_charges'],index=['churn','custserv_calls']))
# 对结果重新格式化， 使用流失情况作为行， 客户服务情况作为 列
# print(churn.pivot_table(['total_charges'],index=['churn'],columns=['custserv_calls']))
# 使用客户服务通话次数作为行 ，流失情况作为列， 指定要计算的 统计量、处理缺失值 和 是否显示 边际值
# print(churn.pivot_table(['total_charges'],index=['custserv_calls'],columns=['churn'],aggfunc='mean',fill_value='NaN',margins=True))


'''3.1 逻辑斯蒂回归'''
'''与 线性回归的区别：'''
# 在这个数据集中， 因变量是一个 二值变量，表示 客户是否已经流失 并不再是 公司客户。
# 线性回归 不适合这种情况， 因为它可能会生成 小于或大于0的 预测结果，在这概率上 没有意义；
# 因为 因变量是一个二值变量，所以需要 将预测值限制在 0和1 之间，；逻辑斯蒂回归 就可以满足要求
'''逻辑斯蒂回归模型'''
# • Pr(yi = 1) = logit-1(β0 +β1xi1 +β2xi2 + … +βpxip)
# 对于 i = 1, 2, …, n 个观测和 p 个输入变量
# 等价于： • Pr(yi = 1) = pi • logit(pi) = (β0 +β1xi1 +β2xi2 + … +βpxip)
# 逻辑斯蒂回归通过使用逻辑函数（或称逻辑斯蒂函数）的  反函数  估计概率的方式来测量自 变量和二值型因变量之间的关系;
#这个函数可以将连续值转换为 0 和 1 之间的值，这是个 必要条件，因为预测值表示概率，而概率必须在 0 和 1 之间;
#这样，逻辑斯蒂回归预测的 就是某种结果的概率，比如客户流失概率。
# 注：逻辑斯蒂回归通过一种能够实现极大似然估计的迭代算法来估计未知的 β 参数值。
# 逻辑斯蒂回归的语法与线性回归有一点区别。对于逻辑斯蒂回归，需要分别设置因变量和 自变量，而不是将它们写在一个公式中：

dependent_variable=churn['churn01']     # 因变量
independent_variables=churn[['account_length','custserv_calls','total_charges']]        # 自变量
# 使用  statsmodels 的 add_constant 函数向输入变量中加入一列 const =1
independent_variables_with_constant=sm.add_constant(independent_variables,prepend=True)
# print(independent_variables_with_constant)
# 拟合 逻辑斯蒂模型
logit_model=sm.Logit(dependent_variable,independent_variables_with_constant).fit()
# 打印摘要信息 ：模型系数、 系数标准差 和 置信区间，、伪 R 方 等模型详细信息。
# print(logit_model.summary())
# 打印出一个列表， 提取出 所有数值信息
# print(dir(logit_model))

# 提取出模型系数和标准差
# print("模型系数：\n",logit_model.params)
# 提取出他们的标准差
# print("标准差：\n",logit_model.bse)

'''3.2 系数解释'''
#1） 对 逻辑斯蒂回归系数的解释 不像线性回归 那么 直观，因为 逻辑斯蒂函数的反函数是条曲线，
# 这说明 自变量 一个 单位的变化 所造成的因变量的变化 不是一个常数。
# 2）因为逻辑斯蒂函数的反函数是 一条曲线， 所以必须选择使用 哪个函数值 来评价自变量对成功概率的影响；
# 和 线性回归一样， 截距系数的意义：当所有自变量为0 时成功的概率。
# 但是有时 0 没有意义，所以另外一种方式 ：当自变量都取均值时， 看看函数的值 有何意义。
def inverse_logit(model_value):
    from math import exp
    return (1.0/(1.0+exp(-model_value)))

# 当所有自变量取均值时 观测的预测值
# 是账户时间、客户服务通话次数和总费用的均 值
at_means=float(logit_model.params[0])+\
float(logit_model.params[1])*float(churn['account_length'].mean())+\
float(logit_model.params[2])*float(churn['custserv_calls'].mean())+\
float(logit_model.params[3])*float(churn['total_charges'].mean())
# print('自变量取均值时的观测值：',round(at_means,3))
# print("逻辑斯蒂函数的反函数在 at_mean 处的值：",round(inverse_logit(at_means),3))
# 所以，当 账户时间、 客户服务通话次数 和 总费用 均取均值时， 客户流失率为 11.2%

# 3）同样，计算某个自变量 一个 单位的变化造成的因变量的变化，
# 可通过 计算当某个自变量从均值发生一个单位的变化时， 成功概率发生了多大的变化

# 计算当 客户服务通话次数 在均值的基础上发生 一个单位的变化时，对客户流失概率造成的影响

cust_serv_mean=float(logit_model.params[0])+\
    float(logit_model.params[1])*float(churn['account_length'].mean())+\
    float(logit_model.params[2])*float(churn['custserv_calls'].mean())+\
    float(logit_model.params[3])*float(churn['total_charges'].mean())
# print('变化之前：',cust_serv_mean)

cust_serv_mean_minus_one=float(logit_model.params[0])+\
    float(logit_model.params[1])*float(churn['account_length'].mean())+\
    float(logit_model.params[2])*float(churn['custserv_calls'].mean()-1.0)+\
    float(logit_model.params[3])*float(churn['total_charges'].mean())
# print("变化后（客户服务通话次数的均值-1）：",cust_serv_mean_minus_one)

# print("两个逻辑斯蒂函数反函数值的差：",inverse_logit(cust_serv_mean)-inverse_logit(cust_serv_mean_minus_one))
# 所以，在均值附近 减少一次 客户通话次数 就对应着 客户流失率 提高 3.7 个百分点

'''3.3 预测'''
# 使用同样的拟合模型 对  新数据  观测进行预测
# 在 churn 数据集中
# 使用 前10个 观测数据创建 10个“新” 观测  （自变量）
new_observations=churn.loc[churn.index.isin(range(10)),independent_variables.columns]
# 使用  statsmodels 的 add_constant 函数向输入变量中加入一列 const =1
new_observations_with_constant=sm.add_constant(new_observations,prepend=True)

# 基于新观测的 账户特性
# 预测客户流失可能性
y_predicted=logit_model.predict(new_observations_with_constant)
print(y_predicted)
# 将预测结果保留两位小数 并打印 ; 从而 通过预测值来评价模型
y_predicted_rounded=[round(score,2) for score in y_predicted ]
print(y_predicted_rounded)




