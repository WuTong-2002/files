"""
火山引擎联网问答Agent - Python后端代理
启动: python server.py
访问: http://localhost:3000
"""
import json
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError
# python server.py
API_URL = "https://open.feedcoopapi.com/agent_api/agent/chat/completion"
API_KEY = "GH1GKRjI4WfpUZwEp6ugp1FX8pJIshHP"
BOT_ID = "7631763725011207721"
PORT = 3000
# python server.py

class Handler(SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/demo.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # 构造请求体
        payload = {
            "bot_id": BOT_ID,
            "messages": body.get("messages", []),
            "stream": body.get("stream", True),
            "extension_options": {
                "enable_processing_state": True,  # 开启工作流状态输出
            },
        }

        # ===== 打印请求 =====
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 发送请求到火山引擎 API")
        print(f"{'='*60}")
        print(f"URL: {API_URL}")
        print(f"Bot ID: {BOT_ID}")
        print(f"Stream: {payload['stream']}")
        print(f"Messages ({len(payload['messages'])} 条):")
        for msg in payload["messages"]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{role}] {preview}")
        print(f"{'-'*60}")

        import time as _time
        _req_start = _time.time()

        req = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )

        try:
            resp = urlopen(req, timeout=60)
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"❌ API 返回错误 ({e.code}): {err_body[:500]}")
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body.encode("utf-8"))
            return

        # ===== 流式透传 + 实时逐帧日志 =====
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 收到响应，开始逐帧输出")
        print(f"{'─'*60}")
        self.send_response(200)
        ct = resp.headers.get("Content-Type", "text/event-stream")
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        buffer = ""
        frame_num = 0
        content_parts = []
        references = []
        follow_ups = []
        usage = None
        tool_calls_list = []

        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
            buffer += chunk.decode("utf-8", errors="replace")

            # 逐行解析 SSE 帧
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()

                if data == "[DONE]":
                    print(f"\n  帧#{frame_num+1} ⏹️  [DONE] 流结束")
                    frame_num += 1
                    continue

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                frame_num += 1
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                delta = obj.get("choices", [{}])[0].get("delta") or {}
                finish = obj.get("choices", [{}])[0].get("finish_reason", "")

                # --- 处理状态帧 ---
                ps = delta.get("processing_state")
                if ps:
                    action = ps.get("action", "")
                    desc = ps.get("description", "")
                    icons = {
                        "planning": "🧠", "search_begin": "🔍",
                        "search_finish": "✅", "tools_begin": "🔧",
                        "tools_finish": "✅", "tools_decision": "💡",
                        "summary_info": "📝",
                    }
                    icon = icons.get(action, "⚙️")
                    print(f"\n  帧#{frame_num} {icon} 处理状态: {action}")
                    print(f"    描述: {desc}")
                    continue

                # --- processing_finish 帧 ---
                if finish == "processing_finish":
                    print(f"\n  帧#{frame_num} ⏩ 处理阶段结束，进入总结生成")
                    continue

                # --- 首帧：参考资料 + 搜索结果 + 卡片 ---
                refs = obj.get("references")
                search = obj.get("search_results")
                cards = obj.get("cards")

                if refs and not references:
                    references = refs
                    print(f"\n  帧#{frame_num} 📚 参考资料 ({len(refs)} 条):")
                    for i, r in enumerate(refs[:10]):
                        src = r.get("source_type", "")
                        title = r.get("title", "")[:50]
                        print(f"    [{i+1}] ({src}) {title}")

                if search:
                    print(f"\n  帧#{frame_num} 🔍 搜索结果 ({len(search)} 条):")
                    for i, s in enumerate(search[:5]):
                        title = s.get("title", "")[:50]
                        src = s.get("source_type", "")
                        print(f"    [{i+1}] ({src}) {title}")
                    if len(search) > 5:
                        print(f"    ... 还有 {len(search)-5} 条")

                if cards:
                    print(f"\n  帧#{frame_num} 🃏 卡片 ({len(cards)} 张):")
                    for c in cards:
                        ctype = c.get("card_type", "")
                        detail = ""
                        if ctype == "video":
                            vc = c.get("video_card", {})
                            detail = f" - {vc.get('title', '')[:40]}"
                        print(f"    - {ctype}{detail}")

                # --- 内容帧 ---
                text = delta.get("content", "")
                if text:
                    content_parts.append(text)
                    # 每帧内容实时打印（短的直接显示，长的截断）
                    display = text.replace("\n", "\\n")
                    if len(display) > 80:
                        display = display[:80] + "..."
                    print(f"  帧#{frame_num} � \"{display}\"", end="")
                    if finish == "stop":
                        print(" [stop]")
                    else:
                        print()

                # --- 工具调用帧 ---
                tc = delta.get("tool_calls")
                if tc:
                    tool_calls_list.extend(tc)
                    for t in tc:
                        fn = t.get("function", {})
                        print(f"\n  帧#{frame_num} 🔧 工具调用: {fn.get('name', '?')}")
                        args = fn.get("arguments", "")
                        if args:
                            print(f"    参数: {args[:200]}")

                # --- 图文混排帧 ---
                img = delta.get("image_info")
                if img:
                    print(f"\n  帧#{frame_num} 🖼️  图片: {img.get('image_url', '')[:60]}")
                    print(f"    尺寸: {img.get('width', '?')}x{img.get('height', '?')}")
                    src_url = img.get("source_url", "")
                    if src_url:
                        print(f"    来源: {src_url[:60]}")

                imgs = delta.get("image_infos")
                if imgs:
                    print(f"\n  帧#{frame_num} �️  多图 ({len(imgs)} 张):")
                    for im in imgs[:3]:
                        print(f"    - {im.get('image_url', '')[:60]}")

                videos = delta.get("video_infos")
                if videos:
                    print(f"\n  帧#{frame_num} 🎬 视频 ({len(videos)} 个):")
                    for v in videos[:3]:
                        print(f"    - {v.get('url', '')[:60]} ({v.get('duration', 0)}ms)")

                # --- 追问帧 ---
                fu = obj.get("follow_ups")
                if fu:
                    follow_ups = fu
                    print(f"\n  帧#{frame_num} 🔄 追问建议:")
                    for f in fu:
                        print(f"    - {f.get('item')}")

                # --- Token 用量帧 ---
                u = obj.get("usage")
                if u:
                    usage = u
                    print(f"\n  帧#{frame_num} 📊 Token: 输入={u.get('prompt_tokens',0)} 输出={u.get('completion_tokens',0)} 总计={u.get('total_tokens',0)}")

        # ===== 汇总 =====
        print(f"\n{'─'*60}")
        full_text = "".join(content_parts)
        print(f"📋 汇总: {frame_num} 帧, 回复 {len(full_text)} 字, 参考 {len(references)} 条, 工具调用 {len(tool_calls_list)} 次")
        if tool_calls_list:
            print(f"  🔧 工具: {', '.join(t.get('function',{}).get('name','?') for t in tool_calls_list)}")
        _req_elapsed = _time.time() - _req_start
        print(f"⏱️  总耗时: {_req_elapsed:.2f}s")
        print(f"{'='*60}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"服务已启动: http://localhost:{PORT}")
    print(f"Bot ID: {BOT_ID}")
    if BOT_ID == "YOUR_BOT_ID":
        print("⚠️  请在 server.py 中替换 BOT_ID 为你的智能体ID")
    server.serve_forever()
