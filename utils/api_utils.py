import os
import time
import requests
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")

class APIClient:
    def __init__(self):
        self.models = {
            "openai": ["gpt-4o-2024-11-20", "gpt-4o", "gpt-4o-mini", "o1-2024-12-17", "o1-mini"],
            # "gemini": "gemini-1.5-flash",
            "deepseek": ["deepseek-chat"],
            "kimi": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "moonshot-v1-auto"],
            "zhipu": ["GLM-4-Plus", "GLM-4-Flash","GLM-4V-Flash"]
        }
        self.max_retries = 3

    def _call_api(self, client, messages, sampling_params, model, stream=False):
        """通用API调用函数，包含重试机制"""
        for attempt in range(self.max_retries):
            try:
                if stream:
                    stream_response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **sampling_params,
                        stream=True
                    )
                    for chunk in stream_response:
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end="")
                    return None # 流式输出不需要返回值
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **sampling_params
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                else:
                    raise  # 达到最大重试次数后抛出异常

    def call_openai(self, messages, sampling_params, model="gpt-4o-mini", stream=False):
        client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")
        return self._call_api(client, messages, sampling_params, model, stream)

    def call_deepseek(self, messages, sampling_params, model="deepseek-chat", stream=False): #添加默认值
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        return self._call_api(client, messages, sampling_params, model, stream)

    def call_kimi(self, messages, sampling_params, model, stream=False):
        messages = [{"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"}]+messages
        client = OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
        return self._call_api(client, messages, sampling_params, model, stream)

    def call_zhipu(self, messages, sampling_params, model, stream=False):
        client = OpenAI(api_key=ZHIPU_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")
        return self._call_api(client, messages, sampling_params, model, stream)

if __name__ == "__main__":
    api_client = APIClient()
    messages = [{"role": "user", "content": "你好"}]
    sampling_params = {"temperature":0.7}
    try:
        response_openai = api_client.call_openai(messages,sampling_params)
        print(f"OpenAI Response: {response_openai}")

        response_zhipu = api_client.call_zhipu(messages,sampling_params,model="GLM-4-Flash")
        print(f"Zhipu Response: {response_zhipu}")

        # response_deepseek = api_client.call_deepseek(messages,sampling_params)
        # print(f"deepseek Response: {response_deepseek}")

        # response_kimi = api_client.call_kimi(messages,sampling_params,model="moonshot-v1-8k")
        # print(f"Kimi Response: {response_kimi}")

    except Exception as e:
        print(f"测试过程中发生异常: {e}")