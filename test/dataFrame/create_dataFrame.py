# coding=utf-8
import pandas as pd

# 通过字典创建DataFrame
dicr = {'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]}
df = pd.DataFrame(dicr)
# df.show() 方法错误 AttributeError: 'DataFrame' object has no attribute 'show'
print(df)

# 先创建list 在创建DataFrame
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
df = pd.DataFrame({'col1': list1, 'col2': list2})
print(df)

# 从list创建DataFrame 指定data 和columns
list1 = [1, 4]
list2 = [5, 8]
df = pd.DataFrame(data=[list1, list2], columns=['age', 'source'])
print(df)

# 修改列名
df.columns = ['Age', 'Source']
print(df)

# 调整DataFrame列顺序, 列名必须一致
df = df[['Source', 'Age']]
print(df)

# 调整index为从1开始
df.index = range(1, len(df) + 1)
print(df)

# DataFrame 使用sql 并不是直接写sql语句!
# https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html
