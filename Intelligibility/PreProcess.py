import os
import re
import csv
import jieba
from gensim import corpora, models, similarities
from collections import Counter

def get_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_files.append(os.path.join(directory, filename))
    return txt_files

# 使用方法
txt_files = get_txt_files('Text')
print(txt_files)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        items = line.strip().split(',')
        row = [item[1:-1] for item in items]
        data.append(row)

    return data

# 使用方法
data = read_file(txt_files[0])
print(data)

def read_original(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for i in range(0, len(lines), 2):
        pair = [lines[i].strip()]
        if i + 1 < len(lines):
            pair.append(lines[i+1].strip())
        data.append(pair)

    return data

# 使用方法
data_original = read_original('OriginalFinal.txt')
print(data_original)


def calculate_proportion(text1, text2):
    counter1 = Counter(text1)
    counter2 = Counter(text2)
    common_chars = counter1 & counter2
    proportion = sum(common_chars.values()) / sum(counter1.values())
    return proportion

def calculate_similarity(text1, text2):
    # 使用正则表达式去掉标点符号和特殊符号
    text1 = re.sub(r'\W', ' ', text1)
    text2 = re.sub(r'\W', ' ', text2)
    dictionary = corpora.Dictionary([list(jieba.cut(text)) for text in [text1, text2]])
    corpus = [dictionary.doc2bow(list(jieba.cut(text))) for text in [text1, text2]]
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2)
    index = similarities.MatrixSimilarity(lda[corpus])
    sims = index[lda[corpus[0]]]
    return sims[1]

for file in txt_files[18:]:
    file_list = read_file(file)
    with open('Output\\'+file,'w') as f:
        for line in file_list:
            print(line[1])
            f.write(line[1])
            f.write(',')
            #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
            #f.write(',')
            f.write(str(((calculate_similarity(line[2],data_original[int(line[0])][0]))+(calculate_similarity(line[3],data_original[int(line[0])][1])))/2))
            f.write('\n')