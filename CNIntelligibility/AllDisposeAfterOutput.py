import os
import pandas as pd

# 初始化一个空的DataFrame来存储结果
result = pd.DataFrame()

# 遍历"Output\Text"文件夹下的所有txt文件
for filename in os.listdir('Output\Text'):
    if filename.endswith('.txt'):
        # 读取txt文件
        file_path = os.path.join('Output\Text', filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        # 处理文件内容，将其转换为DataFrame
        temp_dict = {}
        for line in data:
            parts = line.strip().split(',')
            if len(parts) == 3:
                situation, IntelliTraditional, IntelliLDA = parts
                temp_dict[f'IntelliTraditional_{situation}'] = IntelliTraditional
                temp_dict[f'IntelliLDA_{situation}'] = IntelliLDA

        # 创建一个新的DataFrame，并用文件名作为行标识符
        df = pd.DataFrame([temp_dict], index=[filename.rstrip('.txt')])

        # 将新的DataFrame合并到结果DataFrame中
        result = pd.concat([result, df])

# 将结果DataFrame写入到csv文件中
result.to_csv('mergedIntelligibility.csv')
