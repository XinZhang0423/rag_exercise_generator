from zhipuai import ZhipuAI
api_key="238f2a145d23b824b632c817d3e29436.f6BPfuAd50Lni8Ra"

client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v-flash",  # 填写需要调用的模型名称
    messages=[
        {
          "role": "user", 
          "content": [
            {
              "type": "image_url",
              "image_url": {
                "url" : "sfile.chatglm.cn/testpath/xxxx.jpg"
              }
            },
            {
              "type": "text",
              "text": "图里有什么"
            }
          ]
        },
    ],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta)

