import requests

API_KEY = "your_api_key"

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "什么是扩散模型？"}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.json()["choices"][0]["message"]["content"])
