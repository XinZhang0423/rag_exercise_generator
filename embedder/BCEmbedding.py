import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

BCE_EMBEDDING="/home/dalhxwlyjsuo/Edushare/OpenModels/bce-embedding-base_v1"
BCE_RERANKER="/home/dalhxwlyjsuo/Edushare/OpenModels/bce-reranker-base_v1"

def buildBCEmbedder(text_path, index_path, 
                   chunk_size=2000, overlap=200, 
                   model_name=BCE_EMBEDDING,
                   device='cuda', 
                   batch_size=64, 
                   normalize_embeddings=True, 
                   show_progress_bar=True):
    """
    构建并保存一个 FAISS 向量数据库，用于存储文本块的嵌入向量。

    参数:
        text_path (str): 要嵌入的文本路径。
        index_path (str): 保存 FAISS 索引的路径。
        chunk_size (int, optional): 每个文本块的大小。默认为 1000。
        overlap (int, optional): 相邻文本块之间的重叠大小。默认为 200。
        model_name (str, optional): 使用的预训练嵌入模型的名称。默认为 "sentence-transformers/all-MiniLM-L6-v2"。
        device (str, optional): 运行模型的设备，如 'cuda' 或 'cpu'。默认为 'cuda'。
        batch_size (int, optional): 嵌入模型的批处理大小。默认为 64。
        normalize_embeddings (bool, optional): 是否对嵌入向量进行归一化。默认为 True。
        show_progress_bar (bool, optional): 是否显示嵌入过程的进度条。默认为 True。

    返回:
        FAISS: 包含嵌入向量的 FAISS 向量数据库。
    """
    
    # 1. 将文本分割成块
    

    loader = TextLoader(text_path)
    documents = loader.load()  # 加载文档

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    docs = splitter.split_documents(documents)
    print(f"Text split into {len(docs)} chunks.")
    
    # 2. 初始化HuggingFace嵌入模型
    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'batch_size': batch_size, 
            'normalize_embeddings': normalize_embeddings, 
        }
    )
    print(f"Initialized HuggingFace embedding model: {model_name}")
    
    # 3. 从文本块创建FAISS向量存储
    faiss_vectorstore = FAISS.from_documents(
        documents=docs, 
        embedding=embed_model, 
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    print(f"FAISS vector store created with {faiss_vectorstore.index.ntotal} vectors.")
    
    # 4. 将FAISS索引保存到本地
    faiss_vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}.")
    # faiss_store_intro=FAISS.load_local(index_file_path_intro,embeddings=embed_model,allow_dangerous_deserialization=True)
    # 可选地，加载FAISS索引以验证
    # new_db = FAISS.load_local(index_path, embed_model)
    # print(f"Loaded FAISS index with {new_db.ntotal} vectors.")
    
    return faiss_vectorstore



if __name__=="__main__":
    # sentences = ['sentence_0', 'sentence_1']
    # tokenizer = AutoTokenizer.from_pretrained(BCE_EMBEDDING)
    # model = AutoModel.from_pretrained(BCE_EMBEDDING)

    # device = 'cuda'  # if no GPU, set "cpu"
    # model.to(device)

    # # get inputs
    # inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    # # get embeddings
    # outputs = model(**inputs_on_device, return_dict=True)
    # embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
    # embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
    # print(embeddings)
    
    # tokenizer = AutoTokenizer.from_pretrained(BCE_RERANKER)
    # model = AutoModelForSequenceClassification.from_pretrained(BCE_RERANKER)

    # device = 'cuda'  # if no GPU, set "cpu"
    # model.to(device)

    # # get inputs
    # sentence_pairs=sentences
    # inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    # # calculate scores
    # scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
    # scores = torch.sigmoid(scores)
    # print(scores)
    text_intro_path="/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_image_free.md"
    text_statistic_path="/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_image_free.md"
    # with open(text_intro_path,'r',encoding='utf-8') as f:
    #     text_intro=f.read()
    # print(text_intro)
    index_file_path_intro = "/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro_index_raw"
    index_file_path_statistic = "/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic_index_raw"

    # 指定使用第1号 GPU ('cuda:1')
    faiss_store_intro = buildBCEmbedder(
        text_path=text_intro_path, 
        index_path=index_file_path_intro,
        chunk_size=2000,    # 根据需要调整
        overlap=200,        # 根据需要调整
        model_name=BCE_EMBEDDING,
        device='cuda',    # 指定使用第1号 GPU
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    # print(faiss_store_intro.__dict__)

    index_to_docstore_id = faiss_store_intro.index_to_docstore_id
    print(list(index_to_docstore_id.keys()))
    first_index = list(index_to_docstore_id.keys())[0]
    first_doc_id = index_to_docstore_id[first_index]
    first_doc = faiss_store_intro.get_by_ids([first_doc_id])
    print(first_doc[0].page_content)

    print('-'*200)
    # with open(text_statistic_path,'r',encoding='utf-8') as f:
    #     text_statistic=f.read()
    # print(text_statistic)
        # 指定使用第1号 GPU ('cuda:1')
    # faiss_store_statistic = buildBCEmbedder(
    #     text_path=text_statistic_path, 
    #     index_path=index_file_path_statistic,
    #     chunk_size=2000,    # 根据需要调整
    #     overlap=200,        # 根据需要调整
    #     model_name=BCE_EMBEDDING,
    #     device='cuda',    # 指定使用第1号 GPU
    #     batch_size=64,
    #     normalize_embeddings=True,
    #     show_progress_bar=True
    # )
