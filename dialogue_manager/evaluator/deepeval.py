from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

import torch
import transformers
from transformers import BitsAndBytesConfig
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric

# 传统metrics
class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

class BLEUMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, bleu_type: str = "bleu1"):
        self.threshold = threshold
        self.bleu_type = bleu_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.sentence_bleu_score(
            references=test_case.expected_output,
            prediction=test_case.actual_output,
            bleu_type=self.bleu_type
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return f"{self.bleu_type} BLEU Metric"

class ExactMatchMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.exact_match_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Exact Match Metric"

class QuasiExactMatchMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.quasi_exact_match_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Quasi Exact Match Metric"

class QuasiContainsMetric(BaseMetric):
    """计算准包含指标（判断预测是否准包含于目标列表中的任何一项）。"""
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        if isinstance(test_case.expected_output, str):
            targets = [test_case.expected_output]
        elif isinstance(test_case.expected_output, list):
            targets = test_case.expected_output
        else:
            raise TypeError("Expected output must be a string or a list of strings for QuasiContainsMetric")
        
        self.score = self.scorer.quasi_contains_score(
            targets=targets,
            prediction=test_case.actual_output
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Quasi Contains Metric"

class TruthIdentificationMetric(BaseMetric):
    """计算预测中正确答案的百分比。"""
    def __init__(self, threshold: float = 100.0): #阈值改为100，表示完全正确
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.truth_identification_score(
            target=test_case.expected_output,
            prediction=test_case.actual_output
        )
        self.success = self.score >= self.threshold #只有完全正确才算成功
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Truth Identification Metric"

class SentenceBertMetric(BaseMetric):
    """使用 Sentence-BERT 计算语义相似度。"""

    def __init__(self, threshold: float = 0.7, model_name: str = "all-mpnet-base-v2"):
        """
        初始化 SentenceBertMetric。

        Args:
            threshold (float): 相似度阈值，默认为 0.7。
            model_name (str): Sentence-BERT 模型名称，默认为 "all-mpnet-base-v2"。
        """
        self.threshold = threshold
        self.model_name = model_name
        self.scorer = Scorer() # 这里可以不用Scorer了，直接用sentence_transformers

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        self.model = SentenceTransformer(self.model_name)

    def measure(self, test_case: LLMTestCase):
        """
        测量预测输出和预期输出之间的语义相似度。

        Args:
            test_case (LLMTestCase): 包含实际输出和预期输出的测试用例。

        Returns:
            float: 语义相似度得分。
        """
        embeddings1 = self.model.encode(test_case.actual_output)
        embeddings2 = self.model.encode(test_case.expected_output)
        self.score = util.cos_sim(embeddings1, embeddings2).item()  # 获取标量值
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Sentence BERT Metric"

class AnswerRelevancyMetric(BaseMetric): #补充AnswerRelevancyMetric
    """计算答案相关性得分。"""
    def __init__(self, threshold: float = 0.7, model_type: str = "cross_encoder", model_name: str = None):
        self.threshold = threshold
        self.model_type = model_type
        self.model_name = model_name
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.answer_relevancy_score(
            predictions=test_case.actual_output,
            target=test_case.expected_output,
            model_type=self.model_type,
            model_name = self.model_name
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy Metric"



class CustomLocalModel(DeepEvalBaseLLM):
    def __init__(self,local_model_path):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        
        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"




if __name__=="__main__":
    test_case = LLMTestCase(input="...", actual_output="...", expected_output="...")
    metric = RougeMetric()

    metric.measure(test_case)
    print(metric.is_successful())
    
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    evaluate([test_case], [answer_relevancy_metric])