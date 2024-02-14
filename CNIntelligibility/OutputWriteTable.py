import os
import pandas as pd

def write(method):
    # 初始化一个空的DataFrame来存储结果
    result = pd.DataFrame()

    # 遍历"Output\method\Text"文件夹下的所有txt文件
    for filename in os.listdir('Output/'+method+'/Text'):
        if filename.endswith('.txt'):
            # 读取txt文件
            df = pd.read_csv(os.path.join('Output/'+method+'/Text', filename), header=None, names=['序号', filename.rstrip('.txt')])
            # 将df转换为以'序号'为索引的形式，以便在合并数据时能够匹配相应的序号
            df.set_index('序号', inplace=True)
            # 如果结果DataFrame为空，则直接使用当前df
            if result.empty:
                result = df
            else:
                # 否则，将df合并到结果DataFrame中
                result = result.join(df, how='outer')
    # 将结果DataFrame写入到Excel文件中
    result.to_excel('mergedIntelligibility_'+method+'.xlsx')


write('bm25')
write('difflib')
write('jaccard')
write('lda')
write('levenshtein')
write('simhash')
write('proportion')
