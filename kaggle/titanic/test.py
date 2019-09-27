# -*- coding: utf-8 -*-

import pandas as pd
import exercise_numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

datas = []
for line in open('pre_matrix.txt'):
    datas.append(eval(line))
    # eval(str) 官方解释为：将字符串str当成有效的表达式来求值并返回计算结果。
datas = np.array(datas)
print(datas)

new_data = pd.DataFrame(datas)
# datas.corr()[u'健身'] #只显示“健身”与其他特征的相关系数
# datas[u'健身'].corr(datas[u'教育']) #计算“健身”与“教育”的相关系数
corr = new_data.corr()

# corr.to_csv('corr.txt')

# 保存图片
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, cmap='RdBu', linewidths=0.05, ax=ax)
# 设置Axes的标题
ax.set_title('Correlation between features')
f.savefig('corr.png', dpi=100, bbox_inches='tight')
