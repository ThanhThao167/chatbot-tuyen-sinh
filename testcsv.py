import requests

payload = {
    "session_id": "demo",
    "messages": [
        {"role": "user", "content": "data mining là gì"}
    ]
}

res = requests.post("http://127.0.0.1:8000/chat", json=payload)
print(res.json())
