import pandas as pd

def conbert(method):
    # 读取Excel文件
    df = pd.read_excel('mergedIntelligibility_'+method+'.xlsx')

    # 转置DataFrame
    df_transposed = df.transpose()

    # 将转置后的DataFrame写入到新的Excel文件中
    df_transposed.to_excel('transposedIntelligibility_'+method+'.xlsx', header=False)

conbert('bm25')
conbert('difflib')
conbert('jaccard')
conbert('lda')
conbert('levenshtein')
conbert('simhash')