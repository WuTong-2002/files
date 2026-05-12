from __future__ import annotations

import os
import time
import threading
from typing import Any

from openai import OpenAI

# ============================================================
# 速率限制器：令牌桶算法，15次/分钟
# ============================================================

class RateLimiter:
    """基于令牌桶的速率限制器，线程安全"""

    def __init__(self, max_calls: int = 15, period: float = 60.0):
        self._max_tokens = max_calls
        self._period = period
        self._tokens = max_calls
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        获取一个令牌。如果没有可用令牌，阻塞等待直到有令牌可用。
        返回等待的秒数（0 表示无需等待）。
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill

            # 按时间比例补充令牌
            refill = elapsed * (self._max_tokens / self._period)
            self._tokens = min(self._max_tokens, self._tokens + refill)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # 计算需要等待的时间
            wait_time = (1.0 - self._tokens) * (self._period / self._max_tokens)
            self._tokens = 0.0
            return wait_time

    def wait_and_acquire(self) -> float:
        """等待直到获取令牌，返回实际等待的秒数"""
        wait = self.acquire()
        if wait > 0:
            print(f"  ⏳ [速率限制] 等待 {wait:.1f}s 以避免 API 限流...")
            time.sleep(wait)
            # 等待后重新获取
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                refill = elapsed * (self._max_tokens / self._period)
                self._tokens = min(self._max_tokens, self._tokens + refill)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                else:
                    self._tokens = 0.0
            return wait
        return 0.0


# 全局速率限制器：15次/分钟
_llm_rate_limiter = RateLimiter(max_calls=15, period=60.0)


# ============================================================
# LLM 调用
# ============================================================

def call_llm(
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str | dict[str, Any]:
    """
    统一 LLM 调用入口。

    - 兼容旧用法: 传 prompt（且不传 messages/tools）时返回字符串
    - 工具模式: 传 messages 或 tools 时返回 assistant message 字典
    - 内置速率限制（15次/分钟）和自动重试机制
    """
    client = OpenAI(
        api_key="sk-a9d5f540eac6446ca69563782863f644",
        base_url="https://maas.hikvision.com.cn/v1",
    )

    if messages is not None:
        msgs = list(messages)
    elif prompt is not None:
        msgs = [{"role": "user", "content": prompt}]
    else:
        raise ValueError("Either prompt or messages must be provided")

    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]

    kwargs: dict[str, Any] = {
        "model": "Qwen3.6-35B-A3B-FP8",
        "messages": msgs,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error = None
    for attempt in range(max_retries + 1):
        # 每次调用前等待速率限制令牌
        waited = _llm_rate_limiter.wait_and_acquire()

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower() or "limit" in error_msg.lower():
                print(f"  ⚠️ [API 限流] 第 {attempt + 1}/{max_retries + 1} 次尝试失败: {e}")
            else:
                print(f"  ⚠️ [API 错误] 第 {attempt + 1}/{max_retries + 1} 次尝试失败: {e}")

            if attempt < max_retries:
                delay = retry_delay * (attempt + 1)  # 递增等待: 5s, 10s, 15s
                print(f"  🔄 [重试] 等待 {delay:.0f}s 后重试...")
                time.sleep(delay)
                continue
            else:
                raise RuntimeError(f"LLM API 调用失败，已重试 {max_retries} 次: {last_error}") from last_error

        # 检查响应是否为空
        if response is None or not response.choices:
            last_error = RuntimeError("LLM API 返回空响应，可能是 API 限流或服务不可用")
            print(f"  ⚠️ [空响应] 第 {attempt + 1}/{max_retries + 1} 次尝试返回空响应")

            if attempt < max_retries:
                delay = retry_delay * (attempt + 1)
                print(f"  🔄 [重试] 等待 {delay:.0f}s 后重试...")
                time.sleep(delay)
                continue
            else:
                raise RuntimeError(f"LLM API 返回空响应，已重试 {max_retries} 次") from last_error

        break  # 成功，跳出重试循环

    message = response.choices[0].message

    # 兼容老示例（chatbot/workflow）: 只要是简单 prompt 模式就返回字符串
    if messages is None and tools is None and system_prompt is None:
        return message.content or ""

    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }

    # 提取 token 用量
    if response.usage:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    if message.tool_calls:
        result["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]

    return result


def fake_get_weather(city: str) -> str:
    """模拟天气查询，实际项目中这里会调用真实 API"""
    return f"晴，25°C"


def demo_full_tool_call():
    """演示完整的 tool calling 流程"""
    import json

    # 定义工具
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    }

    # 工具名 -> 实际函数的映射
    tool_functions = {
        "get_weather": fake_get_weather,
    }

    # 对话历史
    messages = [{"role": "user", "content": "北京今天天气怎么样？"}]

    print("=" * 50)
    print("完整 Tool Calling 流程演示")
    print("=" * 50)

    # ---- 第一轮：LLM 决定调用工具 ----
    print("\n【第1轮】用户提问，LLM 返回 tool_call")
    result = call_llm(messages=messages, tools=[weather_tool])
    messages.append(result)  # 把 assistant 消息加入历史

    print(f"  content: {result.get('content')!r}")
    print(f"  tool_calls: {result.get('tool_calls') is not None}")

    if not result.get("tool_calls"):
        print("  LLM 没有调用工具，直接回复了。演示结束。")
        return

    # ---- 执行工具 ----
    for tc in result["tool_calls"]:
        fn_name = tc["function"]["name"]
        fn_args = json.loads(tc["function"]["arguments"])

        print(f"\n【执行工具】{fn_name}({fn_args})")
        tool_result = tool_functions[fn_name](**fn_args)
        print(f"  返回: {tool_result}")

        # 把工具结果作为 tool message 追加到对话
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_result,
        })

    # ---- 第二轮：LLM 基于工具结果生成最终回复 ----
    print("\n【第2轮】把工具结果喂回 LLM，生成最终回复")
    final = call_llm(messages=messages, tools=[weather_tool])
    print(f"  content: {final.get('content')}")
    print(f"  tool_calls: {final.get('tool_calls')}")


if __name__ == "__main__":
    demo_full_tool_call()
