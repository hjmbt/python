# -*- coding:utf-8 -*-
import sys
import pandas as pd
# 官网详情: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
# import seaborn as sns
import matplotlib.pyplot as plt
import exercise_numpy as np

# 解决np print 打印不全问题
np.set_printoptions(threshold=np.inf)
# 显示 列 None-所有列 n-n列
pd.set_option('display.max_columns', None)
# 显示 行 None-所有行 n-n行
pd.set_option('display.max_rows', 4)
# 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 100)

sys.path.append("F:/GitHub/python/kaggle/titanic/")

# 加载数据
test_data = pd.read_csv('./csv/test.csv')
train_data = pd.read_csv('./csv/train.csv')
# DataFrame.set_index(['col','col2']) 使用现有列作为索引 inplace为True时 索引将会还原为列 append添加新索引
# 使用PassengerId作为索引创建新DataFrame
train_data.set_index(['PassengerId'], inplace=True)
test_data.set_index(['PassengerId'], inplace=True)

full_data = train_data.append(test_data)
# 猜测 : 取出DataFrame中的Survived字段
train_y = train_data.Survived
# train_data.describe()
# train_data.replace('male',0,inplace=True)
# train_data.replace('female',1,inplace=True)
# print(train_data.info())

# DataFrame.head(self, n=5)
# Return the first n rows. 返回前n行 默认5行
# print(train_data.head(1))
# pandas 库 pd.DataFrame.corr
# 只显示 Survived 与其他特征的相关系数
# cols = train_data.corr()['Survived']
# 只显示 Survived 与其他特征的相关系数的index
# cols = train_data.corr()['Survived'].index
# print(cols)

# pandas 库 drop(labels,index=0)方法
# 删除 Series 的元素或 DataFrame 的某一行（列）的意思，(0-行 1-列)
full_data = full_data.drop(columns=['Name', 'Survived'])
# print(full_data)

# DataFrame.copy(deep=True) 默认为true
# 复制此对象的索引和数据。
# 当deep=True（默认）时，将使用调用对象的数据和索引的副本创建新对象。
# 对副本的数据或索引的修改不会反映在原始对象中。
# 当deep=False，将创建一个新对象而不复制调用对象的数据或索引（仅复制对数据和索引的引用）。
# 对原始数据的任何更改都将反映在浅层副本中（反之亦然）。
# 返回值 : Series，DataFrame，Panel
full_data_dummy = full_data.copy()
# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
# 将所有的值以1表示,行列拆分,如:
# a | 拆分成 | a_1,a_2
# 1 |        | 1  , 0
# 2 |        | 0  , 1
full_data_dummy = pd.get_dummies(full_data, columns=['Pclass', 'Embarked', 'Sex', 'Ticket'])
# print(full_data_dummy.head(1))
# 删除Cabin这一列
full_data_dummy = full_data_dummy.drop('Cabin', axis=1)
# print(full_data_dummy.isnull().sum().sort_values(ascending=False).head(15))

# fillna() 使用指定的方法填充null值
# mean() 平均值
full_data_dummy = full_data_dummy.fillna(full_data_dummy.mean())

numeric_cols = ['Age', 'Fare']
# loc() 通过标签或布尔数组访问一组行和列。
# 平均值
numeric_col_mean = full_data_dummy.loc[:, numeric_cols].mean()
# std() 返回样本标准差。
# 标准差
numeric_col_std = full_data_dummy.loc[:, numeric_cols].std()
# 数据归一化
# 公式 : n = (n-平均值)/标准差
full_data_dummy.loc[:, numeric_cols] = (full_data_dummy.loc[:, numeric_cols] - numeric_col_mean) / numeric_col_std

# y与x的散点图，具有不同的标记大小和/或颜色。
# plt.ylabel('X')
# plt.ylabel('Y')
plt.scatter(full_data.fillna(full_data.mean()).Age, full_data.Fare)
# plt.show()
# print(plt_scatter)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_data_dummy = full_data_dummy.loc[train_data.index]
test_data_dummy = full_data_dummy.loc[test_data.index]

N_estimators = [20, 50, 100, 150, 200, 250, 300]
test_scores = []
train_X = train_data_dummy
for N in N_estimators:
    # RandomForestRegressor 随机森林回归
    # 官网: https://scikit-learn.org/stable/modules/classes.html
    # n_estimators 随机森林树木的数量
    clf = RandomForestRegressor(n_estimators=N, max_features=0.3)
    # sqrt() 返回数组的非负平方根。
    # cross_val_score 通过交叉验证来评估分数
    test_score = np.sqrt(-cross_val_score(clf, train_X, train_y, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.figure()
plt.title('N_estimator vs CV Error')
plt.plot(N_estimators, test_scores)
plt.show()

rf = RandomForestRegressor(n_estimators=200, max_features=0.3)
rf.fit(train_X, train_y)

# LogisticRegression 逻辑回归分类器
lg = LogisticRegression(C=1.0)
lg.fit(train_X, train_y)

rf_predict = rf.predict(test_data_dummy)
lg_predict = lg.predict_proba(test_data_dummy)[:, 1]
y_final = (rf_predict + lg_predict) / 2
y_final = y_final.round()
y_final = y_final.astype(int)
print(y_final)

submission = pd.DataFrame(data={'PassengerId': test_data.index, 'Survived': y_final})
submission.to_csv('titanic_subm.csv', index=False)
