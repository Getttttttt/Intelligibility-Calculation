import pandas as pd

# 尝试使用不同的编码格式，如 'latin1', 'ISO-8859-1', 或 'cp1252'
try:
    df = pd.read_csv('intelligibility_transformer.csv', index_col=0, encoding='utf-8')  # 假设第一列是行名
except UnicodeDecodeError:
    try:
        df = pd.read_csv('intelligibility_transformer.csv', index_col=0, encoding='latin1')  # 尝试使用latin1编码
    except UnicodeDecodeError:
        df = pd.read_csv('intelligibility_transformer.csv', index_col=0, encoding='ISO-8859-1')  # 尝试使用ISO-8859-1编码

# 对每一行进行Z-score标准化
z_score_df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# 保存到新的CSV文件
z_score_df.to_csv('intelligibility_transformer_z_score.csv', encoding='utf-8')

print('Z-score标准化后的数据已保存到intelligibility_transformer_z_score.csv')
