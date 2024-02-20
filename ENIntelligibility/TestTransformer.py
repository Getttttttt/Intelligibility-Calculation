from sentence_transformers import SentenceTransformer
import numpy as np
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(type(embeddings))

# 计算余弦相似度
dot_product = np.dot(embeddings[0], embeddings[1])
norm_vector1 = np.linalg.norm(embeddings[0])
norm_vector2 = np.linalg.norm(embeddings[1])
cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

print(cosine_similarity)
