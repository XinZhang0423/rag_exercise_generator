from langchain_community.vectorstores import FAISS

class FAISSRecall():
    def __init__(self, faiss_index: FAISS):
        self.faiss_index = faiss_index

    def recall(self, query, k=10):
        return self.faiss_index.similarity_search(query, k=k)
