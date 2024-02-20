import pandas as pd

def addColumnName(method):
    # 读取Excel文件
    df = pd.read_excel('transposedIntelligibility_'+method+'.xlsx')
    name = '_intelligibility_'+method
    # 单独处理第一列，不添加后缀
    first_column_name = df.columns[0]

    # 添加后缀到除第一列外的每个单元格的内容后面
    df.columns = [first_column_name] + [col + name for col in df.columns[1:]]

    # 将修改后的DataFrame写入到新的Excel文件中
    df.to_excel('intelligibility_'+method+'.xlsx', index=False)


# addColumnName('bm25')
# addColumnName('difflib')
# addColumnName('jaccard')
# addColumnName('lda')
# addColumnName('levenshtein')
# addColumnName('simhash')
# addColumnName('proportion')
addColumnName('transformer')