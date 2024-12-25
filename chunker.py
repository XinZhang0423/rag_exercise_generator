import openai
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import FAISS

openai.api_key = 'sk-TFEK1cnYpgPzvcorNthXDVYxsBhXgrT6pcA3KWuPdmvxsWyX'

def split_text(file_path, chunk_size=2000, chunk_overlap=200):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def build_faiss_index(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    if 'openai' in model_name:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    chunk_embeddings = embeddings.embed_documents(chunks)
    embedding_dim = len(chunk_embeddings[0])
    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(np.array(chunk_embeddings).astype('float32'))
    index.add(np.array(chunk_embeddings).astype('float32'))
    return index, embeddings

embedding_model = OpenAIEmbeddings()

faiss_index = FAISS.from_texts(chunks, embedding_model)

def classify_chunk_with_gpt(chunk):
    # 调用GPT-4 Turbo进行分类（你需要调整你的提示语）
    response = openai.Completion.create(
        model="gpt-4-turbo",
        prompt=f"Classify the following scientific content as useful or not:\n\n{chunk}",
        max_tokens=10
    )
    classification = response.choices[0].text.strip()
    return classification

# 对每个数据块进行分类
classified_chunks = []
for chunk in chunks:
    classification = classify_chunk_with_gpt(chunk)
    classified_chunks.append((chunk, classification))

# 输出分类结果
for chunk, classification in classified_chunks:
    print(f"Classification: {classification}")
    print(f"Chunk: {chunk[:200]}...")  # 显示块的前200个字符作为示例
    print("="*100)

# 可选择保存FAISS数据库，以便后续查询
faiss_index.save_local('faiss_index')

