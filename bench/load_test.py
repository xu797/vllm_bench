import asyncio
import time
import argparse
import aiohttp

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

async def one_request(session, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(URL, json=payload) as resp:
        body = await resp.json()
    end = time.perf_counter()
    usage = body.get("usage", {})
    return {
        "latency": end - start,
        "out_tokens": usage.get("completion_tokens", 0),
        "in_tokens": usage.get("prompt_tokens", 0),
    }

async def worker(session, prompt, max_tokens, deadline, stats):
    while time.perf_counter() < deadline:
        try:
            r = await one_request(session, prompt, max_tokens)
            stats.append(r)
        except Exception as e:
            print("request failed:", e)

async def main(concurrency: int, duration: int, max_tokens: int):
    prompt = "请用大约 200 字说明推理服务中吞吐和延迟为什么难以同时最优。"
    stats = []
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        deadline = start + duration
        tasks = [
            asyncio.create_task(worker(session, prompt, max_tokens, deadline, stats))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start

    total_out = sum(s["out_tokens"] for s in stats)
    latencies = sorted(s["latency"] for s in stats)
    def pct(p):
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx] * 1000

    print(f"并发={concurrency} 时长={wall:.1f}s 请求数={len(stats)}")
    print(f"系统吞吐={total_out / wall:.1f} tokens/s")
    print(f"P50={pct(0.50):.0f}ms P95={pct(0.95):.0f}ms P99={pct(0.99):.0f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.duration, args.max_tokens))
