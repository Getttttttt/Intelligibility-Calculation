import os
import re
import csv
import jieba
import jieba.posseg as pseg
import jieba.analyse as als
from gensim import corpora, models, similarities
#import gensim.summarization
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from difflib import SequenceMatcher
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sentence_transformers import SentenceTransformer

def get_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_files.append(os.path.join(directory, filename))
    return txt_files

# 使用方法
txt_files = get_txt_files('Text')
#print(txt_files)

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
#print(data)

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
#print(data_original)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def calculate_similarity_transformer(text1,text2):
    sentences = [text1, text2]
    embeddings = model.encode(sentences)
    dot_product = np.dot(embeddings[0], embeddings[1])
    norm_vector1 = np.linalg.norm(embeddings[0])
    norm_vector2 = np.linalg.norm(embeddings[1])
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity

def calculate_proportion(text1, text2):
    counter1 = Counter(text1)
    counter2 = Counter(text2)
    common_chars = counter1 & counter2
    proportion = sum(common_chars.values()) / sum(counter1.values())
    return proportion

def calculate_similarity_LDA(text1, text2):
    # 使用正则表达式去掉标点符号和特殊符号
    text1 = re.sub(r'\W', ' ', text1)
    text2 = re.sub(r'\W', ' ', text2)
    dictionary = corpora.Dictionary([list(jieba.cut(text)) for text in [text1, text2]])
    corpus = [dictionary.doc2bow(list(jieba.cut(text))) for text in [text1, text2]]
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2)
    index = similarities.MatrixSimilarity(lda[corpus])
    sims = index[lda[corpus[0]]]
    return sims[1]

def calculate_similarity_cos(text1, text2):
    # 使用正则表达式去掉标点符号和特殊符号
    text1 = re.sub(r'\W', ' ', text1)
    text2 = re.sub(r'\W', ' ', text2)
    print(text1)
    print(text2)
    # 将文本转换为词袋模型向量
    vectorizer = CountVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    vectors = vectors.toarray()
    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]
    return similarity_score

def calculate_similarity_tfidf(text1, text2):
    # 使用正则表达式去掉标点符号和特殊符号
    text1 = re.sub(r'\W', ' ', text1)
    text2 = re.sub(r'\W', ' ', text2)
    print(text1)
    print(text2)
    # 创建TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer()
    # 将文本转换为TF-IDF特征矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    print(tfidf_matrix[0])
    print("next")
    print(tfidf_matrix[1])
    # 计算余弦相似度
    similarity_score = cosine_similarity(tfidf_matrix)[0][1]
    print(similarity_score)
    return similarity_score

def calculate_similarity_levenshtein_distance(text1, text2):
    m = len(text1)
    n = len(text2)
    # 创建一个(m+1) x (n+1)的二维数组，并初始化第一行和第一列
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # 动态规划计算编辑距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if text1[i - 1] == text2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # 删除操作
                           dp[i][j - 1] + 1,        # 插入操作
                           dp[i - 1][j - 1] + cost) # 替换操作
    # 归一化编辑距离得分到0-1的范围内
    max_len = max(m, n)
    similarity_score = 1 - dp[m][n] / max_len
    return similarity_score

def calculate_similarity_difflib(text1, text2):
    sequenceMatcher = SequenceMatcher()
    sequenceMatcher.set_seqs(text1, text2)
    return sequenceMatcher.ratio()

def splitWords(str_a):
    wordsa=pseg.cut(str_a)
    cuta = ""
    seta = set()
    for key in wordsa:
        #print(key.word,key.flag)
        cuta += key.word + " "
        seta.add(key.word)
    return [cuta, seta]
def calculate_similarity_jaccard(text1,text2):
    seta = splitWords(text1)[1]
    setb = splitWords(text2)[1]
    sa_sb = 1.0 * len(seta & setb) / len(seta | setb)
    return sa_sb

def calculate_similarity_lsa(text1, text2, n_components=2):
    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # 使用LSA降维
    lsa = TruncatedSVD(n_components=n_components)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    # 计算余弦相似度
    similarity_score = cosine_similarity(lsa_matrix)[0, 1]
    return similarity_score

class SimHash(object):
    def __init__(self):
        jieba.initialize()  # 初始化jieba分词

    def sim_hash(self, content):
        seg = jieba.cut(content)
        key_words = als.extract_tags("|".join(seg), topK=10, withWeight=True)

        key_list = []
        for feature, weight in key_words:
            weight = int(weight)
            bin_str = self.string_hash(feature)
            temp = []
            for c in bin_str:
                if c == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            key_list.append(temp)
        list_sum = np.sum(np.array(key_list), axis=0)
        if key_list == []:
            return '00'
        sim_hash = ''
        for i in list_sum:
            if i > 0:
                sim_hash += '1'
            else:
                sim_hash += '0'

        return sim_hash

    def string_hash(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2**128 - 1
            for c in source:
                x = ((x*m)^ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]

            return str(x)

    def get_distance(self, hash_str1, hash_str2):
        length = 0
        for index, char in enumerate(hash_str1):
            if char == hash_str2[index]:
                continue
            else:
                length += 1

        return length
def calculate_similarity_hash(text1, text2):
    simhash = SimHash()
    hash_str1 = simhash.sim_hash(text1)
    hash_str2 = simhash.sim_hash(text2)
    distance = simhash.get_distance(hash_str1, hash_str2)
    similarity_score = 1 - distance / 64.0  # 计算相似度
    return similarity_score

def calculate_similarity_bm25(text1, text2):
    # 将文本分词
    text1 = re.sub(r'\W', ' ', text1)
    text2 = re.sub(r'\W', ' ', text2)
    tokenized_corpus = [list(jieba.cut(text)) for text in [text1, text2]]    
    print(tokenized_corpus)
    # 训练 BM25 模型
    bm25 = BM25Okapi(tokenized_corpus)
    # 计算文本相似度
    bm25_similarity = bm25.get_scores(tokenized_corpus[1])
    return bm25_similarity[0]  #第二个文本相对于第一个的相似度


# #LDA
# for file in txt_files:
#     file_list = read_file(file)
#     with open('Output\\lda\\'+file,'w') as f:
#         for line in file_list:
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_LDA(line[2],data_original[int(line[0])][0]))+(calculate_similarity_LDA(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')
# #difflib
# for file in txt_files:
#     print(file)
#     file_list = read_file(file)
#     with open('Output\\difflib\\'+file,'w') as f:
#         for line in file_list:
#             print(line)
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_difflib(line[2],data_original[int(line[0])][0]))+(calculate_similarity_difflib(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')
# #hash
# for file in txt_files:
#     file_list = read_file(file)
#     with open('Output\\simhash\\'+file,'w') as f:
#         for line in file_list:
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_hash(line[2],data_original[int(line[0])][0]))+(calculate_similarity_hash(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')
# #levenshtein_distance
# for file in txt_files:
#     file_list = read_file(file)
#     with open('Output\\levenshtein\\'+file,'w') as f:
#         for line in file_list:
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_levenshtein_distance(line[2],data_original[int(line[0])][0]))+(calculate_similarity_levenshtein_distance(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')

# #jaccard
# for file in txt_files:
#     file_list = read_file(file)
#     with open('Output\\jaccard\\'+file,'w') as f:
#         for line in file_list:
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_jaccard(line[2],data_original[int(line[0])][0]))+(calculate_similarity_jaccard(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')
# #bm25
# for file in txt_files:
#     file_list = read_file(file)
#     with open('Output\\bm25\\'+file,'w') as f:
#         for line in file_list:
#             print(line[1])
#             f.write(line[1])
#             f.write(',')
#             #f.write(str(((calculate_proportion(line[2],data_original[int(line[0])][0]))+(calculate_proportion(line[3],data_original[int(line[0])][1])))/2))
#             #f.write(',')
#             f.write(str(((calculate_similarity_bm25(line[2],data_original[int(line[0])][0]))+(calculate_similarity_bm25(line[3],data_original[int(line[0])][1])))/2))
#             f.write('\n')

#transformer
for file in txt_files:
    file_list = read_file(file)
    with open('Output\\transformer\\'+file,'w') as f:
        for line in file_list:
            print(line[1])
            f.write(line[1])
            f.write(',')
            f.write(str(((calculate_similarity_transformer(line[2],data_original[int(line[0])][0]))+(calculate_similarity_transformer(line[3],data_original[int(line[0])][1])))/2))
            f.write('\n')