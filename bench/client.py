import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"

def stream_chat(prompt: str, max_tokens: int = 256):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time = None
    text_parts = []

    with requests.post(BASE_URL, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                text_parts.append(delta)

    end = time.perf_counter()
    output = "".join(text_parts)
    result = {
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "e2e_ms": (end - start) * 1000,
        "output_chars": len(output),
        "preview": output[:60],
    }
    return result

if __name__ == "__main__":
    print(stream_chat("用三句话解释什么是 KV Cache。"))
