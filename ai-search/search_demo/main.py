"""
联网搜索 Agent Demo - Gradio 版
基于 poipoi-agent 框架，模拟火山引擎联网问答工作流：
  意图判断 → 查询改写 → 联网搜索 → 网页抓取 → 总结回复

启动: python search_demo/main.py
访问: http://localhost:3001
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).parent / "poipoi-agent"))

import gradio as gr

from core.llm import call_llm
from core.node import Node, Flow, shared
from tools.builtins.search import search as ddg_search

PORT = 7862
MAX_TURNS = 5  # 最大对话轮数
MAX_MESSAGES = 10  # 滑动窗口大小，5轮对话 = 10条消息
MAX_TOOL_ROUNDS = 10  # 单次请求最大工具调用轮数，防止无限循环


# ============================================================
# 工具定义（供 LLM function calling 使用）
# ============================================================

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网搜索，获取实时信息。当问题涉及最新新闻、实时数据、产品信息等需要联网才能回答的内容时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，应该是优化后的搜索查询",
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_webpage",
        "description": "抓取网页正文内容。当搜索结果的摘要不够详细，需要获取完整网页内容时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的网页 URL",
                },
            },
            "required": ["url"],
        },
    },
}

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取当前日期和时间。当用户询问现在几点、今天日期、当前时间等问题时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，如 Asia/Beijing、America/New_York，默认 Asia/Beijing",
                },
            },
        },
    },
}

ALL_TOOLS = [SEARCH_TOOL, FETCH_TOOL, GET_TIME_TOOL]


# ============================================================
# 工具实现
# ============================================================

def web_search(query: str) -> str:
    """执行联网搜索"""
    try:
        results = ddg_search(query, max_results=8)
        if not results:
            return "未找到相关结果"
        output = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            output.append(f"[{i}] {title}\n    {body}\n    URL: {href}")
        return "\n\n".join(output)
    except Exception as e:
        return f"搜索出错: {e}"


def fetch_webpage(url: str) -> str:
    """抓取网页正文"""
    import urllib.request
    import re

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"抓取失败: {e}"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间"""
    from datetime import datetime, timezone as tz, timedelta

    tz_offsets = {
        "Asia/Shanghai": 8,
        "Asia/Tokyo": 9,
        "America/New_York": -5,
        "America/Los_Angeles": -8,
        "Europe/London": 0,
        "Europe/Paris": 1,
    }
    offset = tz_offsets.get(timezone, 8)
    now = datetime.now(tz(timedelta(hours=offset)))
    sign = "+" if offset >= 0 else ""
    return f"当前时间（{timezone}）: {now.strftime('%Y年%m月%d日 %H:%M:%S')} (UTC{sign}{offset})"


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "get_current_time": get_current_time,
}


# ============================================================
# 工作流节点
# ============================================================

SYSTEM_PROMPT = """你是一个联网搜索助手，拥有以下工具：
- web_search: 联网搜索实时信息
- fetch_webpage: 抓取网页正文
- get_current_time: 获取当前日期和时间

工作原则：
1. 涉及时间、日期的问题 → 必须调用 get_current_time
2. 涉及实时信息（新闻、天气、股价、最新版本、产品动态等）→ 必须调用 web_search
3. 搜索结果摘要不够详细时 → 可以调用 fetch_webpage 获取完整内容
4. 基于工具返回的真实数据回答，不要编造信息
5. 回答用中文，条理清晰，末尾标注参考来源

重要：你不具备任何内置的时间感知能力，必须通过工具获取时间。遇到不确定的事实性问题，优先搜索。"""

REWRITE_SYSTEM_PROMPT = """你是一个搜索查询改写专家。你的任务是将用户问题中的相对时间表达替换为绝对日期，生成更适合搜索引擎的查询关键词。

改写规则：
1. "今日"/"今天" → 替换为当前日期（如"2026年5月11日"）
2. "昨日"/"昨天" → 替换为当前日期前一天
3. "本周"/"这周" → 替换为具体日期范围
4. "最近"/"近期" → 替换为"2026年5月"
5. "最新" → 保留，但补充当前月份

输出格式：只输出改写后的搜索查询词，不要输出任何解释、标点或额外内容。

示例：
用户问：今天有什么热点新闻
当前时间：2026年5月11日
输出：2026年5月11日 热点新闻"""


def sliding_window(messages: list, max_messages: int = MAX_MESSAGES) -> list:
    """滑动窗口截断：保留最后 max_messages 条消息"""
    if len(messages) <= max_messages:
        return messages
    truncated = messages[-max_messages:]
    print(f"  📎 [滑动窗口] {len(messages)} 条消息 → 保留最后 {len(truncated)} 条")
    return truncated


# ============================================================
# 查询改写
# ============================================================

# 需要改写的时间相关关键词
TIME_KEYWORDS = ["今日", "今天", "昨日", "昨天", "本周", "这周", "最近", "近期", "最新", "当前", "现在"]


def _has_time_keyword(text: str) -> bool:
    return any(kw in text for kw in TIME_KEYWORDS)


def rewrite_query(user_message: str) -> str:
    """将用户问题中的相对时间改写为绝对日期，返回改写后的搜索 query。
    先获取当前时间，再用 LLM 做改写。"""
    from datetime import datetime, timedelta

    now = datetime.now()
    today_str = now.strftime("%Y年%m月%d日")
    yesterday_str = (now - timedelta(days=1)).strftime("%Y年%m月%d日")

    rewrite_messages = [
        {"role": "user", "content": f"用户问：{user_message}\n当前时间：{today_str}"},
    ]

    print(f"  📝 [查询改写] 检测到时间关键词，开始改写...")
    t0 = time.time()

    result = call_llm(
        messages=rewrite_messages,
        system_prompt=REWRITE_SYSTEM_PROMPT,
    )

    rewritten = result.get("content", "").strip()
    elapsed = time.time() - t0

    if rewritten and rewritten != user_message:
        print(f"  📝 [查询改写] 原始: \"{user_message}\" → 改写: \"{rewritten}\" ({elapsed:.2f}s)")
        return rewritten
    else:
        print(f"  📝 [查询改写] 无需改写或改写失败，使用原始查询 ({elapsed:.2f}s)")
        return user_message


class RewriteNode(Node):
    """查询改写节点：在意图判断之前，将相对时间改写为绝对日期"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        messages = shared["messages"]
        user_message = messages[-1].get("content", "") if messages else ""

        if _has_time_keyword(user_message):
            rewritten = rewrite_query(user_message)
            shared["rewritten_query"] = rewritten
        else:
            shared["rewritten_query"] = user_message

        return "intent", None


class IntentNode(Node):
    """意图判断节点：LLM 决定是否需要调用工具"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        messages = shared["messages"]
        windowed = sliding_window(messages)

        # 如果有改写后的 query，替换最后一条用户消息的内容
        rewritten = shared.get("rewritten_query", "")
        if rewritten and windowed:
            original = windowed[-1]["content"]
            if rewritten != original:
                windowed = [dict(m) for m in windowed]
                windowed[-1]["content"] = rewritten
                print(f"  🧠 [意图判断] 使用改写后的查询: \"{rewritten[:80]}...\"")

        print(f"\n  🧠 [意图判断] 分析用户问题...")
        t0 = time.time()

        result = call_llm(
            messages=windowed,
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
        elapsed = time.time() - t0
        messages.append(result)
        shared["messages"] = messages

        usage = result.get("usage")
        if usage:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                shared["token_usage"][k] += usage.get(k, 0)
            print(f"  📊 [Token] 本次: 输入={usage.get('prompt_tokens',0)} 输出={usage.get('completion_tokens',0)} 总计={usage.get('total_tokens',0)}")

        if result.get("tool_calls"):
            print(f"  🧠 [意图判断] → 需要调用工具 ({elapsed:.2f}s)")
            return "tool_call", result
        else:
            print(f"  🧠 [意图判断] → 直接回复（无需联网）({elapsed:.2f}s)")
            return "output", result


class ToolCallNode(Node):
    """工具执行节点：执行 LLM 返回的 tool_calls"""
    _call_count = 0
    _round_count = 0  # 工具调用轮数（每次从 intent → tool_call 算一轮）

    def exec(self, payload: Any) -> Tuple[str, Any]:
        response = payload
        messages = shared["messages"]
        tool_calls = response.get("tool_calls", [])
        ToolCallNode._call_count += len(tool_calls)
        ToolCallNode._round_count += 1

        # 检查是否超过最大工具调用轮数
        if ToolCallNode._round_count > MAX_TOOL_ROUNDS:
            print(f"\n  ⚠️ [工具调用] 已达到最大轮数 {MAX_TOOL_ROUNDS}，强制终止工具调用")
            # 构造一个强制输出的消息
            force_msg = {
                "role": "assistant",
                "content": "抱歉，搜索过程过于复杂，已达到最大搜索轮数限制。请尝试简化您的问题，或换个方式提问。",
            }
            messages.append(force_msg)
            shared["messages"] = messages
            return "output", force_msg

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            # 容错解析 arguments：可能是 JSON 字符串或已经是 dict
            if isinstance(raw_args, str):
                try:
                    fn_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    print(f"  ⚠️ [参数解析失败] {fn_name}: 无法解析 arguments: {raw_args[:100]}")
                    fn_args = {}
            elif isinstance(raw_args, dict):
                fn_args = raw_args
            else:
                print(f"  ⚠️ [参数类型错误] {fn_name}: arguments 类型为 {type(raw_args)}")
                fn_args = {}

            # 如果 fn_args 中有错误的 "arguments" 键（LLM 有时会这样返回），尝试修复
            if "arguments" in fn_args and fn_name in TOOL_FUNCTIONS:
                import inspect
                sig = inspect.signature(TOOL_FUNCTIONS[fn_name])
                expected_params = list(sig.parameters.keys())
                # 如果期望的参数不在 fn_args 中，尝试从 "arguments" 值中提取
                if not any(p in fn_args for p in expected_params if p != "arguments"):
                    inner = fn_args["arguments"]
                    if isinstance(inner, str):
                        try:
                            fn_args = json.loads(inner)
                        except json.JSONDecodeError:
                            fn_args = {expected_params[0]: inner} if expected_params else {}
                    elif isinstance(inner, dict):
                        fn_args = inner

            print(f"\n  🔧 [工具调用] {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                t0 = time.time()
                try:
                    result = fn(**fn_args)
                except TypeError as e:
                    print(f"  ⚠️ [参数不匹配] {fn_name}: {e}")
                    # 尝试用第一个参数名传递整个 fn_args 的值
                    import inspect
                    sig = inspect.signature(fn)
                    params = list(sig.parameters.keys())
                    if params and fn_args:
                        first_param = params[0]
                        # 如果只有一个参数且 fn_args 有值，尝试直接传
                        if len(params) == 1 and len(fn_args) == 1:
                            val = list(fn_args.values())[0]
                            result = fn(**{first_param: val})
                        else:
                            result = f"工具调用参数错误: {e}"
                    else:
                        result = f"工具调用参数错误: {e}"
                elapsed = time.time() - t0
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  📄 [工具结果] ({elapsed:.2f}s) {preview}")
            else:
                result = f"未知工具: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        shared["messages"] = messages
        return "intent", None


class OutputNode(Node):
    """输出节点：收集最终回复"""
    _last_content = ""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        response = payload
        content = response.get("content", "")
        OutputNode._last_content = content
        print(f"\n{'─'*60}")
        print(f"🤖 {content[:100]}...")
        print(f"{'─'*60}")
        return "default", content


# ============================================================
# 核心处理函数
# ============================================================

def process_message(messages: list) -> Tuple[str, dict]:
    """处理一条用户消息，返回 (回复内容, token用量)"""
    from datetime import datetime

    user_message = messages[-1].get("content", "") if messages else ""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 开始处理: {user_message[:50]}...")
    print(f"  消息数: {len(messages)}")
    print(f"{'='*60}")

    # 初始化状态
    shared.clear()
    shared["messages"] = list(messages)
    shared["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 构建工作流
    rewrite = RewriteNode()
    intent = IntentNode(max_retries=8, wait=15)
    tool_call = ToolCallNode()
    output = OutputNode()
    rewrite - "intent" >> intent
    intent - "tool_call" >> tool_call
    tool_call - "intent" >> intent
    intent - "output" >> output

    # 重置计数器
    ToolCallNode._call_count = 0
    ToolCallNode._round_count = 0
    OutputNode._last_content = ""

    t_start = time.time()

    # 运行工作流
    try:
        flow = Flow(rewrite)
        flow.run(None)
    except Exception as e:
        print(f"\n  ❌ [错误] 工作流执行失败: {e}")
        full_content = f"抱歉，处理您的请求时遇到了错误：{e}。请稍后重试或简化问题。"
        token = shared["token_usage"]
        return full_content, token

    full_content = OutputNode._last_content
    tool_calls_count = ToolCallNode._call_count
    t_total = time.time() - t_start
    token = shared["token_usage"]

    print(f"\n{'─'*60}")
    print(f"📋 汇总: 回复 {len(full_content)} 字, 工具调用 {tool_calls_count} 次")
    print(f"⏱️  总耗时: {t_total:.2f}s")
    print(f"📊 Token: 输入={token['prompt_tokens']} 输出={token['completion_tokens']} 总计={token['total_tokens']}")
    print(f"{'='*60}\n")

    return full_content, token


# ============================================================
# Gradio 聊天函数
# ============================================================

def chat(
    message: str,
    history: list,
) -> tuple:
    """Gradio 聊天函数，支持多轮对话。history 格式为 [{"role": "user", "content": "..."}, ...]"""

    if not message.strip():
        return "", history

    # history 已经是 OpenAI 格式的字典列表，直接追加当前用户消息
    messages = list(history)  # 复制一份
    messages.append({"role": "user", "content": message})

    # 截断到最大轮数
    if len(messages) > MAX_TURNS * 2:
        messages = messages[-(MAX_TURNS * 2):]

    # 调用后端处理
    reply, _ = process_message(messages)

    # 更新历史（追加 assistant 回复）
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return "", history


def clear_history() -> tuple:
    """清空对话历史"""
    return "", []


# ============================================================
# Gradio 界面
# ============================================================

def build_ui():
    with gr.Blocks(title="联网问答 Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 联网问答 Agent
        基于本地 Agent 框架，支持实时联网搜索和多轮对话
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_copy_button=True,
                    type="messages",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="输入你的问题... (Shift+Enter 换行，Enter 发送)",
                        lines=3,
                        scale=4,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    gr.Examples(
                        examples=[
                            ["今天有什么热点新闻"],
                            ["推荐一些好看的电影"],
                            ["Python和Go哪个更适合后端开发"],
                            ["现在几点了？"],
                            ["杭州和北京明天哪个更热"],
                            ["杭州附近有哪些美食"],
                            ["周杰伦最新的消息"],
                            ["你先帮我看一下几点了，然后查看一下杭州天气，再看一下周杰伦的消息"],
                        ],
                        inputs=msg_input,
                        label="示例问题",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 功能说明")
                gr.Markdown("""
                - 🔍 **联网搜索**: 实时获取最新信息
                - 🌐 **网页抓取**: 获取完整网页内容
                - ⏰ **时间查询**: 获取当前时间
                - 💬 **多轮对话**: 支持5轮上下文
                """)
                gr.Markdown("### 技术栈")
                gr.Markdown("""
                - 框架: poipoi-agent
                - 前端: Gradio
                - 搜索: DuckDuckGo
                - LLM: Qwen3.5
                """)

        # 事件绑定 - chat 返回 (清空后的输入框, 更新后的历史)
        submit_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, msg_input],
        )

    return demo


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("🔍 联网搜索 Agent（Gradio 版）")
    print("=" * 60)
    print(f"启动服务: http://localhost:{PORT}")
    print("工作流: 意图判断 → 工具调用(搜索/抓取) → 总结回复")
    print("按 Ctrl+C 停止服务\n")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)


if __name__ == "__main__":
    main()
