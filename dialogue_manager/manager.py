# modules/dialogue_manager/manager.py

import os
import random

from recall.faiss_recall import FAISSRecall
from recall.tfidf_recall import TfidfRecall
from recall.bm25_recall import BM25Recall
from langchain_community.vectorstores import FAISS
from embedder.BCEmbedding import BCE_EMBEDDING,BCE_RERANKER,buildBCEmbedder
from utils.api_utils import APIClient
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from prompt_assembler import PromptsRegistry

class DialogueManager:
    def __init__(self, config):
        self.config = config

        # Initialize Embedder
        self.embed_model = HuggingFaceEmbeddings(
            model_name=BCE_EMBEDDING,
            model_kwargs={'device':'cuda'},
            encode_kwargs={
                'batch_size':config['embedding']['batch_size'],
                'normalize_embeddings':config['embedding']['normalize_embeddings']
            }
        )
        
        # Load FAISS index for passages
        if config['recall']['method'] == "faiss":
            self.chunk_faiss_vectorstore = buildBCEmbedder(text_path=config['faiss']['chunk_text_path'],
                             index_path=config['faiss']['chunk_index_path'],
                             overlap=config['faiss']['overlap'],    
                             chunk_size=config['faiss']['chunk_size'],    
                             model_name=BCE_EMBEDDING)
            self.chunk_recall_module = FAISSRecall(self.chunk_faiss_vectorstore)
            
            # self.exercise_faiss_vectorstore = FAISS.load_local(config['faiss']['exercise_index_path'], self.embed_model, allow_dangerous_deserialization=True)
            # self.exercise_recall_module = FAISSRecall(self.exercise_faiss_vectorstore)
        
    def load_random_chunk(self,idx=None):
        """
        随机抽取一个chunk
        """
        # 获取 FAISS 索引的总数
        index_to_docstore_id = self.chunk_faiss_vectorstore.index_to_docstore_id
        total_vectors = len(list(index_to_docstore_id.keys()))
        print(f'random selecting from {total_vectors} passages')
        # 随机选择一个向量的 ID
        if idx is None:
            random_id = list(index_to_docstore_id.keys())[random.randint(0, total_vectors - 1)]
        else:
            random_id = list(index_to_docstore_id.keys())[idx]

        first_doc_id = index_to_docstore_id[random_id]
        first_doc = self.chunk_faiss_vectorstore.get_by_ids([first_doc_id])

        
        # print(first_doc)
        if first_doc:
            # 假设返回的第一个文档是我们需要的
            random_text = first_doc[0].page_content  # 文本内容
            random_vector = self.chunk_faiss_vectorstore.index.reconstruct(random_id)  # 对应的向量
            return random_text, random_vector
        else:
            print('some error occurs in random selection')
            return None, None
    
    def is_knowledge_intensive(self, chunk):
        """
        判断chunk是否是知识密集型的
        """
        example_useful = PromptsRegistry.assemble('EXAMPLES_USEFUL_zh')
        example_useless = PromptsRegistry.assemble('EXAMPLES_USELESS_zh')
        format_instructions = "参考示例只输出是否，以及原因"

        # 组装 CLEAN_PROMPT
        clean_prompt = PromptsRegistry.assemble(
            'CLEAN_PROMPT_zh',
            example_useful=example_useful,
            example_useless=example_useless,
            chunk=chunk,
            format_instructions=format_instructions
        )
        
        client = APIClient()
        messages=[{'role': 'user', 'content': clean_prompt}]
        SamplingParams={'temperature': 0, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        if isinstance(response, str):
            if "是" in response:
                is_intensive=True
                reason=response.split('因为')[-1].strip()
            else:
                is_intensive=False
                reason=response.split('因为')[-1].strip()
        else:
            is_intensive=False
            reason='error!'
        return is_intensive,reason
    
    def label_chunk(self, chunk,topic='计算机导论'):
        """
        标注chunk的类别、子类别和知识点
        """
        label_chunk_prompt = PromptsRegistry.assemble(
            'LABEL_CHUNK_PROMPT_zh',
            chunk=chunk,
        )
        client = APIClient()
        messages=[{'role': 'user', 'content': label_chunk_prompt}]
        SamplingParams={'temperature': 0, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        if isinstance(response, str):
            return response
        else:
            return 'error'
    
    def recall_similar_chunks(self, chunk, k=10):
        """
        使用FAISS召回相似的chunks
        """
        top_k_chunks = self.chunk_recall_module.recall(chunk, k=k)
        return top_k_chunks
    
    def recall_top10_exercises(self, chunk, k=10):
        """
        使用FAISS召回相似的习题
        """
        top_k_exercises = self.exercise_faiss_index.similarity_search(chunk, k=k)
        return top_k_exercises
    
    def determine_strategies(self, chunk):
        """
        判断适用的出题策略
        """
        reasoning_prompt = PromptsRegistry.assemble(
            'REASONING_PROMPT_zh',
            text=chunk,
        )
        client = APIClient()
        messages=[{'role': 'user', 'content': reasoning_prompt}]
        SamplingParams={'temperature': 0, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        if isinstance(response, str):
            return response
        else:
            return 'error'
       
    def rerank_chunks_for_strategy(self, chunk, strategy, recalled_chunks, recalled_exercises, top_n=3):
        """
        重新排序并选择top3相关的chunks
        """
        # 合并chunks和exercises
        combined_chunks = recalled_chunks + recalled_exercises
        # 使用rerank_module对combined_chunks进行重新排序
        top_n_chunks = self.rerank_module.rerank(chunk,strategy,combined_chunks, top_n=top_n)
        return top_n_chunks
    
    def generate_question_simple(self,chunk,strategy):
        generator=PromptsRegistry.assemble(
            strategy,
            text=chunk
        )
        client = APIClient()
        messages=[{'role': 'user', 'content': generator}]
        SamplingParams={'temperature': 0.9, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        if isinstance(response, str):
            return response
        else:
            return 'error'
       
    def generate_question(self, chunk, strategy, top3_chunks):
        """
        生成题目
        """
        # 构建生成题目的prompt
        prompt = f"为以下内容标注类别、子类别和知识点，格式为：类别, 子类别, 知识点。\n\n内容：{chunk}"
        client = APIClient()
        messages=[{'role': 'User', 'content': prompt}]
        SamplingParams={'temperature': 0.7, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        try:
            result = response.choices[0].text.strip().lower()
        except Exception as e:
            print(e)
            
        return result
    
    def evaluate_question(self, question):
        evaluator="""请根据以下生成的题目，从可读性、适切性、复杂性和参与度四个维度对其质量进行评估。每个维度的评分范围为0到5分，并请具体说明每个分数对应的程度。最后以JSON格式输出评估结果，格式如下：
        {
            "可读性": {
                "得分": 分数,
                "说明": "具体说明"
            },
            "适切性": {
                "得分": 分数,
                "说明": "具体说明"
            },
            "复杂性": {
                "得分": 分数,
                "说明": "具体说明"
            },
        }

        以下是评分维度的详细说明：

        1. **可读性**：评估题目是否易于阅读和理解。
        - 0分：题目非常难以理解，存在严重的语法或用词错误。
        - 1分：题目理解起来非常困难，表达不清晰。
        - 2分：题目有一定难度，部分学生可能难以理解。
        - 3分：题目基本清晰，大多数学生能够理解。
        - 4分：题目表达清晰，易于理解，适合大多数学生。
        - 5分：题目极其清晰，语言简洁明了，非常易于理解。

        2. **学科适宜性**：评估题目在语义上是否与相应学科对齐，符合教学目标。
        - 0分：题目与学科内容完全不相关。
        - 1分：题目与学科内容关联极弱，几乎没有相关性。
        - 2分：题目与学科内容关联较弱，相关性不强。
        - 3分：题目与学科内容基本相关，符合教学目标。
        - 4分：题目与学科内容高度相关，完全符合教学目标。
        - 5分：题目与学科内容完美对齐，极大地支持教学目标。

        3. **复杂性**：评估题目需要的推理或认知努力程度。
        - 0分：题目完全不具备认知挑战，过于简单。
        - 1分：题目几乎没有认知挑战，难度极低。
        - 2分：题目有轻微的认知挑战，难度较低。
        - 3分：题目具备适度的认知挑战，难度适中。
        - 4分：题目具有较高的认知挑战，难度较大。
        - 5分：题目极具认知挑战，难度很高，适合高水平学生。
        """+f"""
        请根据以上维度和说明，对以下生成的题目进行评估，并按照指定的JSON格式输出结果。
        {question}
        """
        client = APIClient()
        messages=[{'role': 'user', 'content': evaluator}]
        SamplingParams={'temperature': 0.8, 'top_p': 0.9, 'presence_penalty': 1.0, 'frequency_penalty': 1.0}
        response = client.call_zhipu(
            model="GLM-4-flash",
            messages=messages,
            sampling_params=SamplingParams
        )
        if isinstance(response, str):
            return response
        else:
            return 'error'
    
    def execute_process_1(self):
        """
        执行完整的题目生成流程
        """
        # Step 1: 随机抽取一个chunk
        random_chunk = self.load_random_chunk()
        
        # Step 2: 判断是否知识密集型
        if not self.is_knowledge_intensive(random_chunk):
            
            return
        
        # Step 3: 标注chunk
        labels = self.label_chunk(random_chunk)
        
        # Step 4: 召回相似chunks和习题
        top_k_chunks = self.recall_similar_chunks(random_chunk, k=10)
        top_k_exercises = self.recall_top10_exercises(random_chunk, k=10)
        
        # Step 5: 判断适用的策略
        applicable_strategies = self.determine_strategies(random_chunk)
        
        for strategy in applicable_strategies:
            # Step 6: 重新排序并选择top3
            top3_chunks = self.rerank_chunks_for_strategy(random_chunk, strategy, top_k_chunks, top_k_exercises, top_n=3)
            
            # Step 7: 生成题目
            question = self.generate_question(random_chunk, strategy, top3_chunks)
            
            # Step 8: 评估题目质量
            score = self.evaluate_question(question)
            
            # 打印或保存结果
            print(f"Strategy: {strategy}\nQuestion: {question}\nScore: {score}\n")
            self.logger.info(f"Generated Question with Strategy {strategy}: {question} (Score: {score})")
