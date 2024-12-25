from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TfidfRecall():
    def __init__(self, all_passages, max_features=5000):
        """
        初始化 TF-IDF 召回模块。

        参数:
        - all_passages (list of str): 所有文本块。
        - max_features (int): TF-IDF 向量的最大特征数。
        """
        self.all_passages = all_passages
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(all_passages)
        print(f"TF-IDF 向量化完成，形状为 {self.tfidf_matrix.shape}。")

    def recall(self, query, k=10):
        """
        使用 TF-IDF 进行召回。

        参数:
        - query (str): 查询文本。
        - k (int): 返回的相似文本块数量。

        返回:
        - top_k_chunks (list of dict): top-k 相似的文本块及其分数。
        """
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        top_k_chunks = [{'page_content': self.all_passages[idx], 'score': cosine_similarities[idx]} for idx in top_k_indices]
        return top_k_chunks
