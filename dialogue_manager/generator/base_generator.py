import os
import random
from langchain_huggingface import HuggingFaceEmbeddings  # 使用最新的导入路径
from langchain.vectorstores import FAISS, DistanceStrategy
from sentence_transformers import CrossEncoder
import torch

def buildBCEmbedder(text, index_path, 
                   chunk_size=1000, overlap=200, 
                   model_name="sentence-transformers/all-MiniLM-L6-v2",
                   device='cuda' if torch.cuda.is_available() else 'cpu', 
                   batch_size=64, 
                   normalize_embeddings=True):
    """
    使用 HuggingFace embeddings 构建 FAISS 向量存储。
    """
    
    # 1. 分割文本
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    passages = splitter.split_text(text)
    print(f"文本已分割为 {len(passages)} 个块。")

    # 2. 初始化 HuggingFace 嵌入模型，不启用量化
    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'batch_size': batch_size, 
            'normalize_embeddings': normalize_embeddings, 
            # 'show_progress_bar': False,  # 已移除以避免冲突
        }
    )
    print(f"已初始化 HuggingFace 嵌入模型: {model_name}")

    # 3. 使用 FAISS 创建向量存储
    faiss_vectorstore = FAISS.from_texts(
        texts=passages, 
        embedding=embed_model, 
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    print(f"FAISS 向量存储已创建，包含 {faiss_vectorstore.index.ntotal} 个向量。")

    # 4. 保存 FAISS 索引
    faiss_vectorstore.save_local(index_path)
    print(f"FAISS 索引已保存到 {index_path}。")

    return faiss_vectorstore

def load_faiss_index(index_path, embedding_model):
    """
    加载已保存的 FAISS 索引。
    
    参数:
    - index_path (str): FAISS 索引的保存路径。
    - embedding_model (HuggingFaceEmbeddings): 用于嵌入的模型。
    
    返回:
    - faiss_vectorstore (FAISS): 加载的 FAISS 向量存储。
    """
    faiss_vectorstore = FAISS.load_local(index_path, embedding_model)
    print(f"已加载 FAISS 索引，包含 {faiss_vectorstore.index.ntotal} 个向量。")
    return faiss_vectorstore

def recall(faiss_vectorstore, all_passages, k=10):
    """
    随机选择一个文本块，并使用 FAISS 找回 top-k 相似的文本块。
    
    参数:
    - faiss_vectorstore (FAISS): FAISS 向量存储。
    - all_passages (list of str): 所有文本块。
    - k (int): 返回的相似文本块数量。
    
    返回:
    - query_chunk (str): 随机选择的查询文本块。
    - top_k_chunks (list of dict): top-k 相似的文本块及其分数。
    """
    query_chunk = random.choice(all_passages)
    print(f"随机选择的查询文本块: {query_chunk[:100]}...")  # 打印前100个字符

    # 使用 FAISS 进行相似性搜索
    top_k_chunks = faiss_vectorstore.similarity_search(query_chunk, k=k)
    print(f"找回了 {len(top_k_chunks)} 个相似的文本块。")
    return query_chunk, top_k_chunks

def rerank(query_chunk, top_k_chunks, rerank_model_name="cross-encoder/stsb-roberta-large", top_n=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用跨编码器模型对 top-k 文本块进行重新排序，选择 top-n 最相关的文本块。
    
    参数:
    - query_chunk (str): 查询文本块。
    - top_k_chunks (list of dict): top-k 相似的文本块及其分数。
    - rerank_model_name (str): 重新排序模型的名称。
    - top_n (int): 返回的最相关文本块数量。
    - device (str): 使用的设备（'cuda' 或 'cpu'）。
    
    返回:
    - top_n_chunks (list of dict): top-n 重新排序后的文本块及其分数。
    """
    # 初始化跨编码器模型
    cross_encoder = CrossEncoder(rerank_model_name, device=device)
    print(f"已加载重新排序模型: {rerank_model_name} 使用设备: {device}")

    # 准备输入对
    pairs = [(query_chunk, chunk['page_content']) for chunk in top_k_chunks]
    
    # 获取分数
    scores = cross_encoder.predict(pairs)
    
    # 将分数与文本块对应
    for i, chunk in enumerate(top_k_chunks):
        chunk['score'] = scores[i]
    
    # 按分数排序，降序
    top_k_chunks_sorted = sorted(top_k_chunks, key=lambda x: x['score'], reverse=True)
    
    # 选择 top-n
    top_n_chunks = top_k_chunks_sorted[:top_n]
    print(f"重新排序后选择了 {len(top_n_chunks)} 个最相关的文本块。")
    
    return top_n_chunks

# 示例用法
if __name__ == "__main__":
    import torch
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # 设置环境变量（根据需要调整）
    # os.environ["BNB_CUDA_VERSION"] = "124"
    # os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/cuda/lib64"

    # 定义模型路径和索引路径
    embedding_model_path = "/home/dalhxwlyjsuo/Edushare/OpenModels/bce-embedding-base_v1"
    index_file_path = "faiss_index"

    # 初始化嵌入模型
    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={
            'batch_size': 64, 
            'normalize_embeddings': True, 
            # 'show_progress_bar': False,  # 已移除
        }
    )

    # 加载 FAISS 索引
    faiss_store = load_faiss_index(index_file_path, embed_model)

    # 获取所有文本块（假设您在构建索引时有保存所有的 passages）
    # 这里假设 passages 被保存为一个列表
    # 如果 passages 不是列表，请根据实际情况调整
    # 例如，如果 passages 是通过 FAISS 向量存储创建的，您需要另外保存这些文本块
    # 这里假设您有一个名为 'all_passages.txt' 的文件，每行一个文本块
    with open("all_passages.txt", "r", encoding="utf-8") as f:
        all_passages = [line.strip() for line in f if line.strip()]
    print(f"加载了 {len(all_passages)} 个文本块。")

    # 进行召回
    query, top_k = recall(faiss_store, all_passages, k=10)

    # 进行重新排序
    top_n = rerank(query, top_k, rerank_model_name="cross-encoder/stsb-roberta-large", top_n=3)

    # 打印结果
    print("Top-3 重新排序后的相关文本块：")
    for idx, chunk in enumerate(top_n, 1):
        print(f"Rank {idx}: {chunk['page_content']} (Score: {chunk['score']})\n")
