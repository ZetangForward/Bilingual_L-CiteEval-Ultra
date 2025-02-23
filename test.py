
import requests

# response = requests.post(
#     "http://127.0.0.1:4100/generate",  
#     # headers={"Content-Type": "application/json"},
#     json={
#         "model": "/nvme/big_model/DeepSeek-V3",
#         "messages": [
#             {"role": "user", "content": "用Python实现快速排序"}
#         ],
#         "max_tokens": 1024,
#         "temperature": 0.7
#     }
# )

# print(response.json())


# from openai import OpenAI

# client = OpenAI(
#     base_url="http://127.0.0.1:4100/", 
#     api_key="EMPTY"  
# )
# response = client.chat.completions.create(
#     model="/nvme/big_model/DeepSeek-V3",  # 必须与启动时的--served-model-name参数一致
#     messages=[
#         {"role": "user", "content": "请解释量子计算的基本原理"}
#     ],
#     max_tokens=512,
#     temperature=0.3
# )

# print(response.choices[0].message.content)


import openai

# client = openai.OpenAI(base_url="http://127.0.0.1:4100/v1/" ,api_key="EMPTY")
# response = client.chat.completions.create(
#     model="deepseek_v3",
#     # prompt = 'hello!',
#     messages=[{"role": "user", "content": "Hello"}],
#     stream=True
# )

# print(response)

import requests

url = "http://127.0.0.1:4100/generate"  # vLLM 服务器地址
# url = 'http://127.0.0.1:4100/v1/chat/completions'
headers = {"Content-Type": "application/json"}

data = {
    "prompt": "Hello, how are you?",
    # "model": "deepseek_v3",
    # 'messages':[{'role':'user',"content":'how are you ?'}],
    "max_tokens": 100
}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # 解析返回的 JSON
