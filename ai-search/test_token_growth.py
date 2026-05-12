"""
测试火山引擎联网问答Agent是否对历史消息做了压缩
方法：用相同的固定消息，逐步增加对话轮数，观察 input token 增长是否线性
"""
import json
import time
from urllib.request import Request, urlopen

API_URL = "https://open.feedcoopapi.com/agent_api/agent/chat/completion"
API_KEY = "GH1GKRjI4WfpUZwEp6ugp1FX8pJIshHP"
BOT_ID = "7631763725011207721"

# 固定的对话模板（每轮内容相同，方便计算预期 token）
ROUND_TEMPLATE = [
    {"role": "user", "content": "请用一句话介绍一下Python语言的特点"},
    {"role": "assistant", "content": "Python是一种简洁易读、功能强大的高级编程语言，以其丰富的库生态和广泛的应用场景著称。"},
]


def test_with_rounds(num_rounds: int) -> dict:
    """发送指定轮数的对话，返回 token 用量"""
    messages = []
    for _ in range(num_rounds):
        messages.extend(ROUND_TEMPLATE)
    # 最后加一个新问题
    messages.append({"role": "user", "content": "你好"})

    payload = {
        "bot_id": BOT_ID,
        "messages": messages,
        "stream": False,  # 非流式，方便直接拿 usage
    }

    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    t0 = time.time()
    resp = urlopen(req, timeout=30)
    elapsed = time.time() - t0
    body = json.loads(resp.read().decode("utf-8"))

    usage = body.get("usage", {})
    return {
        "rounds": num_rounds,
        "msg_count": len(messages),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "elapsed": round(elapsed, 2),
    }


def main():
    print("=" * 70)
    print("火山引擎 Agent 历史消息压缩测试")
    print("=" * 70)
    print(f"每轮对话: user(约30字) + assistant(约40字)")
    print(f"最后追加: user '你好'")
    print(f"注意: API 文档说最多保留最后 10 条消息")
    print()

    # 测试 0, 1, 2, 3, 4, 5 轮历史（即 1, 3, 5, 7, 9, 11 条消息）
    test_rounds = [0, 1, 2, 3, 4, 5]

    results = []
    for n in test_rounds:
        print(f"测试 {n} 轮历史 ({n*2+1} 条消息)...", end=" ", flush=True)
        try:
            r = test_with_rounds(n)
            results.append(r)
            print(f"✅ prompt={r['prompt_tokens']}, completion={r['completion_tokens']}, total={r['total_tokens']}, time={r['elapsed']}s")
        except Exception as e:
            print(f"❌ {e}")
        time.sleep(1)  # 避免限流

    # 汇总表格
    print(f"\n{'='*70}")
    print(f"{'轮数':>4} | {'消息数':>6} | {'input token':>12} | {'增量':>8} | {'耗时':>6}")
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")

    prev_tokens = 0
    for r in results:
        delta = r["prompt_tokens"] - prev_tokens if prev_tokens > 0 else "-"
        delta_str = str(delta) if isinstance(delta, int) else delta
        print(f"{r['rounds']:>4} | {r['msg_count']:>6} | {r['prompt_tokens']:>12} | {delta_str:>8} | {r['elapsed']:>5}s")
        prev_tokens = r["prompt_tokens"]

    # 分析
    if len(results) >= 3:
        print(f"\n{'='*70}")
        print("分析:")
        # 计算每轮增量
        deltas = []
        for i in range(1, len(results)):
            d = results[i]["prompt_tokens"] - results[i-1]["prompt_tokens"]
            deltas.append(d)
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        print(f"  每增加1轮对话，平均增加 {avg_delta:.0f} input token")

        # 检查是否线性
        if len(deltas) >= 3:
            max_d = max(deltas)
            min_d = min(deltas)
            if max_d > 0 and (max_d - min_d) / max_d < 0.2:
                print(f"  增量波动小 ({min_d}~{max_d})，接近线性增长 → 未做压缩")
            else:
                print(f"  增量波动大 ({min_d}~{max_d})，可能存在压缩或截断")

        # 检查 10 条消息截断
        over_10 = [r for r in results if r["msg_count"] > 10]
        under_10 = [r for r in results if r["msg_count"] <= 10]
        if over_10 and under_10:
            last_under = under_10[-1]
            first_over = over_10[0]
            growth = first_over["prompt_tokens"] - last_under["prompt_tokens"]
            print(f"  跨越10条消息边界: {last_under['msg_count']}条→{first_over['msg_count']}条, token增量={growth}")
            if growth < avg_delta * 0.5:
                print(f"  增量明显下降 → 确认存在10条消息截断")


if __name__ == "__main__":
    main()
