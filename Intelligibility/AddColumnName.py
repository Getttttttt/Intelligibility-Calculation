import pandas as pd

# 读取Excel文件
df = pd.read_excel('transposedIntelligibility.xlsx')

# 添加后缀到第一行的每个单元格的内容后面
df.columns = [col + '_intelligibility_LDA' for col in df.columns]

# 将修改后的DataFrame写入到新的Excel文件中
df.to_excel('intelligibilityLDA.xlsx', index=False)
