import exercise_numpy as np
import pandas as pd

file_ia = '''./csv/ia.csv'''
ia = pd.read_csv(file_ia)

# 输出数据的统计信息，包含无缺数据量，最大值、最小值、均值、标准差，四分位数，下面将会详细介绍。
print(ia.describe())
