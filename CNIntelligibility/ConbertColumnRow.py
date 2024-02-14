import pandas as pd

# 读取Excel文件
df = pd.read_excel('mergedIntelligibility.xlsx')

# 转置DataFrame
df_transposed = df.transpose()

# 将转置后的DataFrame写入到新的Excel文件中
df_transposed.to_excel('transposedIntelligibility.xlsx', index=False)
