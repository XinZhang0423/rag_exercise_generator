from rank_bm25 import BM25Okapi
import numpy as np

class BM25Recall():
    def __init__(self, all_passages):
        """
        初始化 BM25 召回模块。

        参数:
        - all_passages (list of str): 所有文本块。
        """
        self.all_passages = all_passages
        tokenized_passages = [passage.split(" ") for passage in all_passages]
        self.bm25 = BM25Okapi(tokenized_passages)
        print("BM25 模型已初始化。")

    def recall(self, query, k=10):
        """
        使用 BM25 进行召回。

        参数:
        - query (str): 查询文本。
        - k (int): 返回的相似文本块数量。

        返回:
        - top_k_chunks (list of dict): top-k 相似的文本块及其分数。
        """
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_chunks = [{'page_content': self.all_passages[idx], 'score': scores[idx]} for idx in top_k_indices]
        return top_k_chunks
