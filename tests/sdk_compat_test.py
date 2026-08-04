"""SDK 合规测试：用官方 anthropic / openai SDK 端到端验证 fake-vlm。

SDK 是最严格的"客户端"——按规范解析所有字段和事件序列，SDK 能过 =
绝大多数真实客户端能过。覆盖非流式/流式/图片/工具/多轮/错误处理。
"""
import base64
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = "fk-local-dev-8f3a2c91"
MODEL = "deepseek-v4-flash"

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name} {detail}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


def load_test_image() -> str:
    data = urllib.request.urlopen(
        "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg",
        timeout=30,
    ).read()
    return base64.b64encode(data).decode()


# ---------- Anthropic SDK ----------
from anthropic import Anthropic

ac = Anthropic(api_key=KEY, base_url=BASE)
img_b64 = load_test_image()

print("\n=== Anthropic SDK 测试 ===")

# 1. 非流式基础对话
try:
    r = ac.messages.create(model=MODEL, max_tokens=200, messages=[{"role": "user", "content": "你好"}])
    check("1 非流式对话", r.type == "message" and r.content and r.content[0].type == "text",
          f"stop={r.stop_reason} usage={r.usage}")
except Exception as e:
    check("1 非流式对话", False, str(e)[:200])

# 2. 流式对话（SDK 事件解析）
try:
    events = []
    with ac.messages.stream(model=MODEL, max_tokens=200, messages=[{"role": "user", "content": "说三个字"}]) as stream:
        for evt in stream:
            events.append(evt.type)
        text = stream.get_final_text()
    ok_seq = "message_start" in events and "content_block_start" in events and "content_block_stop" in events and "message_stop" in events
    check("2 流式对话", ok_seq and len(text) > 0, f"events={events} text='{text[:20]}'")
except Exception as e:
    check("2 流式对话", False, str(e)[:200])

# 3. 图片输入（base64 image 块）
try:
    r = ac.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": "图里是什么题目？"},
    ]}])
    txt = r.content[0].text
    check("3 图片输入", "长方体" in txt or "正方体" in txt or "数学" in txt or "表面积" in txt, f"'{txt[:40]}'")
except Exception as e:
    check("3 图片输入", False, str(e)[:200])

# 4. 工具调用
try:
    r = ac.messages.create(model=MODEL, max_tokens=200, tools=[{
        "name": "get_weather", "description": "查天气",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }], messages=[{"role": "user", "content": "广州天气，用工具"}])
    tu = [c for c in r.content if c.type == "tool_use"]
    check("4 工具调用", len(tu) == 1 and tu[0].name == "get_weather" and isinstance(tu[0].input, dict),
          f"stop={r.stop_reason} tool={tu[0].name if tu else '无'}")
except Exception as e:
    check("4 工具调用", False, str(e)[:200])

# 5. 多轮 tool_result（含纯 image tool_result）
try:
    r = ac.messages.create(model=MODEL, max_tokens=1500, messages=[
        {"role": "user", "content": "看这张图"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "read_1", "name": "Read", "input": {"file_path": "1.png"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_1", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        ]}]},
    ])
    txt = r.content[0].text
    check("5 纯图tool_result", "数学" in txt or "长方体" in txt or "正方体" in txt, f"'{txt[:40]}'")
except Exception as e:
    check("5 纯图tool_result", False, str(e)[:200])

# 6. 多轮工具往返（tool_result 文本 → 继续）
try:
    r = ac.messages.create(model=MODEL, max_tokens=200, tools=[{
        "name": "get_weather", "description": "查天气",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }], messages=[
        {"role": "user", "content": "广州天气，用工具"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "广州"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "28度多云"}]},
        {"role": "user", "content": "要带伞吗"},
    ])
    types = [c.type for c in r.content]
    check("6 工具往返", len(r.content) > 0 and (types[0] == "text" or "tool_use" in types),
          f"blocks={types} text='{r.content[0].text[:40] if types[0]=='text' else ''}'")
except Exception as e:
    check("6 工具往返", False, str(e)[:200])

# 7. 鉴权错误
try:
    Anthropic(api_key="wrong-key", base_url=BASE).messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    check("7 错误key→401", False, "没抛错")
except Exception as e:
    check("7 错误key→401", "401" in str(e) or "authentication" in str(e).lower() or "401" in repr(e), f"{type(e).__name__}")

# ---------- OpenAI SDK ----------
from openai import OpenAI

oc = OpenAI(api_key=KEY, base_url=f"{BASE}/v1")

print("\n=== OpenAI SDK 测试 ===")

# 8. 非流式
try:
    r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "你好"}], max_tokens=200)
    check("8 非流式", r.choices[0].message.content and len(r.choices[0].message.content) > 0, f"model={r.model}")
except Exception as e:
    check("8 非流式", False, str(e)[:200])

# 9. 流式（max_tokens 需覆盖思考+回答，200 会被思考吃光）
try:
    chunks = [c for c in oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "说三个字"}], max_tokens=500, stream=True)]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    check("9 流式", len(text) > 0 and chunks[-1].choices[0].finish_reason == "stop", f"'{text[:20]}' finish={chunks[-1].choices[0].finish_reason}")
except Exception as e:
    check("9 流式", False, str(e)[:200])

# 10. 图片输入（image_url data URL）
try:
    r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": "图里是什么？"},
    ]}], max_tokens=300)
    txt = r.choices[0].message.content
    check("10 图片输入", "数学" in txt or "长方体" in txt or "正方体" in txt, f"'{txt[:40]}'")
except Exception as e:
    check("10 图片输入", False, str(e)[:200])

# 11. 工具调用
try:
    r = oc.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "广州天气，用工具"}], max_tokens=200,
        tools=[{"type": "function", "function": {"name": "get_weather", "description": "查天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    tc = r.choices[0].message.tool_calls
    check("11 工具调用", tc and tc[0].function.name == "get_weather", f"tool={tc[0].function.name if tc else '无'}")
except Exception as e:
    check("11 工具调用", False, str(e)[:200])

# 12. 多轮工具往返
try:
    r = oc.chat.completions.create(model=MODEL, messages=[
        {"role": "user", "content": "广州天气，用工具"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"广州\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "28度多云"},
        {"role": "user", "content": "要带伞吗"},
    ], max_tokens=200)
    check("12 工具往返", len(r.choices[0].message.content) > 0, f"'{r.choices[0].message.content[:40]}'")
except Exception as e:
    check("12 工具往返", False, str(e)[:200])

# 13. 鉴权错误
try:
    OpenAI(api_key="wrong-key", base_url=f"{BASE}/v1").chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])
    check("13 错误key→401", False, "没抛错")
except Exception as e:
    check("13 错误key→401", "401" in str(e), f"{type(e).__name__}")

print(f"\n===== 结果: {PASS} 通过 / {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)
