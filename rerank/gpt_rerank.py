import openai

class Gptrerank():
    def __init__(self, model="gpt-4", api_key=None):
        openai.api_key = api_key
        self.model = model

    def rerank(self, query, candidates, top_n=3):
        # 使用 GPT-4 对候选进行重新排序
        # 示例逻辑：发送查询和候选到 GPT-4，获取排序结果
        pass
