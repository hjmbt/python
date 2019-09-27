import pandas as pd

# 解决np print 打印不全问题
# np.set_printoptions(threshold=np.inf)
# 显示 列 None-所有列 n-n列
pd.set_option('display.max_columns', None)
# 显示 行 None-所有行 n-n行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 100)

filePath_melb_data = '''./csv/melb_data.csv'''
melb_data = pd.read_csv(filePath_melb_data)
print(melb_data.describe())

