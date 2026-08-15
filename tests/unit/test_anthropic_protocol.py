"""anthropic_protocol.py 单元测试：请求解析（Anthropic → OpenAI）与响应转换（OpenAI → Anthropic）。

用例全部来自真实踩过的坑（见 git log），重点回归两个已修复高危 bug：
- anthropic_sse 混合流块索引全局递增（曾文本块与工具块都是 index 0，Anthropic SDK 解析失败）
- anthropic_sse 尾 chunk 的 content_block_stop 必须晚于所有 content_block_delta（曾 stop 先于
  尾块 delta，流式末尾丢字）

纯函数级测试，无网络 / 无真实 key；chunk 用 types.SimpleNamespace 构造。
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from src.anthropic_protocol import (
    _assistant_content_to_openai,
    anthropic_sse,
    parse_messages,
    parse_system,
    parse_tool_choice,
    to_anthropic_message,
)


# ---------- SSE chunk 构造辅助 ----------

def _chunk(cid="chunk_1", content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        id=cid,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


def _tc(index, tid, name, arguments=None):
    return SimpleNamespace(
        index=index, id=tid, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _usage(prompt=12, completion=7, cache_hit=0):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_cache_hit_tokens=cache_hit
    )


async def _iter_chunks(chunks):
    for c in chunks:
        yield c


def _collect(chunks, vision_prefix=None):
    """把 anthropic_sse 输出收集为 [(event_name, data_dict), ...]"""

    async def run():
        events = []
        async for line in anthropic_sse(_iter_chunks(chunks), "deepseek-v4-flash-vl", vision_prefix):
            name_line, _, data_line = line.partition("\n")
            name = name_line.split(": ", 1)[1].strip()
            events.append((name, json.loads(data_line.split(": ", 1)[1])))
        return events

    return asyncio.run(run())


# ---------- parse_messages ----------

class TestParseMessages:
    def test_tool_result_block_becomes_tool_message(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_01ABC",
                "content": "文件内容",
            }],
        }]
        out = parse_messages(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "tool"
        assert out[0]["tool_call_id"] == "toolu_01ABC"
        assert out[0]["content"] == "文件内容"

    def test_tool_result_pure_image_kept_as_image_url(self):
        # Claude Code Read 工具返回纯 image 块——必须保留为 image_url，
        # 路由层才能提取进视觉管线
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_02XYZ",
                "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAACg=="},
                }],
            }],
        }]
        out = parse_messages(msgs)
        assert out[0]["role"] == "tool"
        assert out[0]["tool_call_id"] == "toolu_02XYZ"
        assert out[0]["content"] == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAACg=="}}
        ]

    def test_tool_result_url_image_kept_as_image_url(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_03",
                "content": [{"type": "image", "source": {"type": "url", "url": "http://img/x.png"}}],
            }],
        }]
        out = parse_messages(msgs)
        assert out[0]["content"] == [
            {"type": "image_url", "image_url": {"url": "http://img/x.png"}}
        ]

    def test_text_plus_tool_result_split_into_user_then_tool(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "看下这个文件"},
                {"type": "tool_result", "tool_use_id": "toolu_04", "content": "ok"},
            ],
        }]
        out = parse_messages(msgs)
        assert [m["role"] for m in out] == ["user", "tool"]
        assert out[0]["content"] == [{"type": "text", "text": "看下这个文件"}]
        assert out[1]["tool_call_id"] == "toolu_04"

    def test_assistant_tool_use_then_user_text_then_tool_result_order(self):
        # Claude Code 形态：assistant tool_use → user 纯文本 → user tool_result，
        # 输出顺序必须保持 assistant → user → tool
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_05", "name": "Read", "input": {"path": "/tmp/a.png"}}
            ]},
            {"role": "user", "content": "继续"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_05", "content": "图片已读取"}
            ]},
        ]
        out = parse_messages(msgs)
        assert [m["role"] for m in out] == ["assistant", "user", "tool"]
        assert out[0]["tool_calls"][0]["function"]["name"] == "Read"
        assert out[2]["role"] == "tool"
        assert out[2]["tool_call_id"] == "toolu_05"

    def test_assistant_mixed_text_and_tool_use_blocks(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我来查"},
                {"type": "tool_use", "id": "toolu_06", "name": "get_weather", "input": {"city": "广州"}},
            ],
        }]
        out = parse_messages(msgs)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == [{"type": "text", "text": "我来查"}]
        tc = out[0]["tool_calls"][0]
        assert tc["id"] == "toolu_06"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "广州"}

    def test_assistant_thinking_text_becomes_reasoning_content(self):
        # Claude Code 回传 thinking 块 → 转为 deepseek 的 reasoning_content
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "VLM 描述：图中有一只猫"},
                {"type": "text", "text": "看到一只猫"},
            ],
        }]
        out = parse_messages(msgs)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == [{"type": "text", "text": "看到一只猫"}]
        assert out[0]["reasoning_content"] == "VLM 描述：图中有一只猫"

    def test_assistant_thinking_with_tool_use_keeps_reasoning_content(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "需要查询天气"},
                {"type": "tool_use", "id": "toolu_07", "name": "get_weather", "input": {"city": "北京"}},
            ],
        }]
        out = parse_messages(msgs)
        assert out[0]["role"] == "assistant"
        assert out[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert out[0]["reasoning_content"] == "需要查询天气"

    def test_assistant_multiple_thinking_blocks_joined(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "第一段"},
                {"type": "thinking", "thinking": "第二段"},
                {"type": "text", "text": "结论"},
            ],
        }]
        out = parse_messages(msgs)
        assert out[0]["reasoning_content"] == "第一段\n\n第二段"

    def test_assistant_no_thinking_block_has_no_reasoning_content(self):
        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "无思考"}]}]
        out = parse_messages(msgs)
        assert "reasoning_content" not in out[0]

    def test_unsupported_role_raises(self):
        with pytest.raises(ValueError):
            parse_messages([{"role": "robot", "content": "hi"}])


# ---------- _assistant_content_to_openai ----------

class TestAssistantContentToOpenai:
    def test_thinking_blocks_returned_as_thinking_text(self):
        parts, tool_calls, thinking = _assistant_content_to_openai([
            {"type": "thinking", "thinking": "描述A"},
            {"type": "text", "text": "正文"},
            {"type": "thinking", "thinking": "描述B"},
        ])
        assert parts == [{"type": "text", "text": "正文"}]
        assert tool_calls == []
        assert thinking == "描述A\n\n描述B"

    def test_no_thinking_blocks_returns_empty_string(self):
        parts, tool_calls, thinking = _assistant_content_to_openai([
            {"type": "text", "text": "正文"},
        ])
        assert parts == [{"type": "text", "text": "正文"}]
        assert tool_calls == []
        assert thinking == ""

    def test_string_content_returns_empty_thinking(self):
        parts, tool_calls, thinking = _assistant_content_to_openai("纯文本")
        assert parts == [{"type": "text", "text": "纯文本"}]
        assert tool_calls == []
        assert thinking == ""

    def test_empty_thinking_block_contributes_nothing(self):
        _, _, thinking = _assistant_content_to_openai([
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "正文"},
        ])
        assert thinking == ""


# ---------- parse_system ----------

class TestParseSystem:
    def test_string_form(self):
        assert parse_system("你是助手") == "你是助手"

    def test_list_of_text_blocks(self):
        blocks = [{"type": "text", "text": "规则1"}, {"type": "text", "text": "规则2"}]
        assert parse_system(blocks) == "规则1\n规则2"

    def test_none_returns_empty(self):
        assert parse_system(None) == ""

    def test_non_text_blocks_skipped(self):
        blocks = [{"type": "text", "text": "规则"}, {"type": "image", "source": {}}]
        assert parse_system(blocks) == "规则"


# ---------- parse_tool_choice ----------

class TestParseToolChoice:
    def test_auto_and_none(self):
        assert parse_tool_choice(None) == "auto"
        assert parse_tool_choice("auto") == "auto"
        assert parse_tool_choice("none") == "none"

    def test_any_maps_to_required(self):
        assert parse_tool_choice("any") == "required"

    def test_dict_type_tool(self):
        assert parse_tool_choice({"type": "tool", "name": "get_weather"}) == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_unknown_form_falls_back_auto(self):
        assert parse_tool_choice({"type": "weird"}) == "auto"


# ---------- to_anthropic_message ----------

class TestToAnthropicMessage:
    def _resp(self, content=None, tool_calls=None, finish_reason="stop", usage=None):
        return {
            "id": "chatcmpl_1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_text_content_usage_and_model(self):
        out = to_anthropic_message(self._resp(content="你好"), "deepseek-v4-flash-vl")
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "deepseek-v4-flash-vl"
        assert out["content"] == [{"type": "text", "text": "你好"}]
        assert out["usage"] == {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0}
        assert out["stop_sequence"] is None

    def test_tool_calls_mapped_to_tool_use(self):
        resp = self._resp(
            content="查一下",
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "广州"}'},
            }],
        )
        out = to_anthropic_message(resp, "m")
        assert out["content"] == [
            {"type": "text", "text": "查一下"},
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "广州"}},
        ]
        assert out["stop_reason"] == "tool_use"

    def test_stop_reason_mapping(self):
        assert to_anthropic_message(self._resp(content="a", finish_reason="stop"), "m")["stop_reason"] == "end_turn"
        assert to_anthropic_message(self._resp(content="a", finish_reason="tool_calls"), "m")["stop_reason"] == "tool_use"
        assert to_anthropic_message(self._resp(content="a", finish_reason="length"), "m")["stop_reason"] == "max_tokens"
        assert to_anthropic_message(self._resp(content="a", finish_reason=None), "m")["stop_reason"] == "end_turn"

    def test_invalid_arguments_json_becomes_empty_input(self):
        resp = self._resp(
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_2", "type": "function",
                "function": {"name": "f", "arguments": "{not json"},
            }],
        )
        out = to_anthropic_message(resp, "m")
        assert out["content"] == [{"type": "tool_use", "id": "call_2", "name": "f", "input": {}}]

    def test_vision_prefix_injects_thinking_block_first(self):
        out = to_anthropic_message(
            self._resp(content="你好"), "m", vision_prefix="VLM 描述"
        )
        assert out["content"][0]["type"] == "thinking"
        assert out["content"][0]["thinking"] == "VLM 描述"
        assert "signature" in out["content"][0]  # 带 signature，Claude Code 才会回传
        assert out["content"][1] == {"type": "text", "text": "你好"}

    def test_vision_prefix_with_tool_calls_keeps_thinking_first(self):
        out = to_anthropic_message(
            self._resp(
                content="查一下",
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "广州"}'},
                }],
            ),
            "m",
            vision_prefix="VLM 描述",
        )
        assert out["content"][0]["type"] == "thinking"
        assert out["content"][0]["thinking"] == "VLM 描述"
        assert "signature" in out["content"][0]
        assert out["content"][1]["type"] == "text"
        assert out["content"][2]["type"] == "tool_use"

    def test_no_vision_prefix_has_no_thinking_block(self):
        out = to_anthropic_message(self._resp(content="你好"), "m")
        assert all(b["type"] != "thinking" for b in out["content"])

    def test_empty_vision_prefix_has_no_thinking_block(self):
        out = to_anthropic_message(self._resp(content="你好"), "m", vision_prefix="")
        assert all(b["type"] != "thinking" for b in out["content"])


# ---------- anthropic_sse ----------

class TestAnthropicSse:
    def test_pure_text_stream_event_sequence(self):
        events = _collect([
            _chunk("chunk_1", content="你好"),
            _chunk("chunk_2", content="，世界", finish_reason="stop", usage=_usage()),
        ])
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        # message_start 携带初始消息骨架
        assert events[0][1]["message"]["model"] == "deepseek-v4-flash-vl"
        # 文本块 index 0
        assert events[1][1]["index"] == 0
        assert events[1][1]["content_block"]["type"] == "text"
        assert events[2][1]["delta"]["type"] == "text_delta"
        assert events[2][1]["delta"]["text"] == "你好"
        assert events[3][1]["delta"]["text"] == "，世界"
        assert events[4][1]["index"] == 0
        # stop → end_turn；usage 从尾 chunk 搬运到 message_delta
        assert events[5][1]["delta"]["stop_reason"] == "end_turn"
        assert events[5][1]["usage"] == {"input_tokens": 12, "output_tokens": 7, "cache_read_input_tokens": 0}

    def test_tool_stream(self):
        events = _collect([
            _chunk("chunk_1", tool_calls=[_tc(0, "call_0", "get_weather", '{"city": "')]),
            _chunk("chunk_2", tool_calls=[_tc(0, "call_0", "get_weather", '广州"}')]),
            _chunk("chunk_3", finish_reason="tool_calls", usage=_usage()),
        ])
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        block = events[1][1]
        assert block["index"] == 0
        assert block["content_block"]["type"] == "tool_use"
        assert block["content_block"]["id"] == "call_0"
        assert block["content_block"]["name"] == "get_weather"
        assert events[2][1]["delta"]["type"] == "input_json_delta"
        assert events[2][1]["delta"]["partial_json"] == '{"city": "'
        assert events[3][1]["delta"]["partial_json"] == '广州"}'
        assert events[4][1]["index"] == 0
        assert events[5][1]["delta"]["stop_reason"] == "tool_use"

    def test_mixed_text_and_tool_indices_unique(self):
        # 已修复高危 bug 回归：文本块与工具块曾共用 index 0，
        # 混合流下 Anthropic SDK 解析失败
        events = _collect([
            _chunk("chunk_1", content="我来查", tool_calls=[_tc(0, "call_0", "get_weather", "")]),
            _chunk("chunk_2", tool_calls=[_tc(0, "call_0", "get_weather", '{"city": "广州"}')]),
            _chunk("chunk_3", finish_reason="tool_calls", usage=_usage()),
        ])
        starts = [e[1] for e in events if e[0] == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1]
        assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
        stops = [e[1] for e in events if e[0] == "content_block_stop"]
        assert [s["index"] for s in stops] == [0, 1]
        # 全部 delta 都在 stop 之前
        delta_idxs = [e[1]["index"] for e in events if e[0] == "content_block_delta"]
        assert delta_idxs == [0, 1]
        assert events[-2][1]["delta"]["stop_reason"] == "tool_use"

    def test_finish_reason_after_last_delta_in_same_chunk(self):
        # 已修复高危 bug 回归：finish_reason 曾先于尾块 delta 处理，
        # content_block_stop 抢在 delta 之前发出，流式末尾丢字
        events = _collect([
            _chunk("chunk_1", content="结尾文字", finish_reason="stop", usage=_usage()),
        ])
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2][1]["delta"]["text"] == "结尾文字"
        assert events[3][1]["index"] == 0

    def test_empty_stream_no_hang(self):
        events = _collect([])
        assert [e[0] for e in events] == ["message_start", "message_delta", "message_stop"]
        assert events[0][1]["message"]["id"] == "msg_empty"
        assert events[1][1]["delta"]["stop_reason"] == "end_turn"

    def test_cache_hit_tokens_passthrough_to_message_delta(self):
        # 缓存命中数透传（deepseek prompt_cache_hit_tokens -> Anthropic cache_read_input_tokens），
        # 供 Claude Code 侧监控缓存命中率。
        events = _collect([
            _chunk("chunk_1", content="hi"),
            _chunk("chunk_2", finish_reason="stop", usage=_usage(100, 50, cache_hit=384)),
        ])
        assert events[-2][1]["usage"] == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 384,
        }

    def test_reasoning_content_stream_emits_thinking_block(self):
        # 修复回归：deepseek thinking 模式的 reasoning_content 必须转成 Anthropic thinking 块，
        # 否则下一轮回传空 reasoning，缓存前缀对不上（命中率暴跌）。
        def _reasoning_chunk(cid, reasoning=None, content=None, finish_reason=None):
            delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)
            return SimpleNamespace(id=cid, choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)

        events = _collect([
            _reasoning_chunk("c1", reasoning="思考一"),
            _reasoning_chunk("c2", reasoning="思考二"),
            _reasoning_chunk("c3", content="答案", finish_reason="stop", ),
            _chunk("c4", finish_reason="stop", usage=_usage()),
        ])
        # thinking 块应排在 text 块之前
        starts = [e[1] for e in events if e[0] == "content_block_start"]
        assert [s["content_block"]["type"] for s in starts] == ["thinking", "text"]
        assert [s["index"] for s in starts] == [0, 1]
        # thinking 内容按顺序拼接
        thinking_deltas = [
            e[1]["delta"]["thinking"]
            for e in events if e[0] == "content_block_delta"
            and e[1]["delta"].get("type") == "thinking_delta"
        ]
        assert "".join(thinking_deltas) == "思考一思考二"
        # thinking 块有独立的 content_block_stop
        stops = [e[1]["index"] for e in events if e[0] == "content_block_stop"]
        assert stops == [0, 1]

    def test_three_tool_calls_get_three_blocks_with_unique_indices(self):
        events = _collect([
            _chunk("chunk_1", tool_calls=[
                _tc(0, "call_0", "get_weather", '{"city":'),
                _tc(1, "call_1", "get_stock", '{"code":'),
                _tc(2, "call_2", "get_time", '{"tz":'),
            ]),
            _chunk("chunk_2", tool_calls=[
                _tc(0, "call_0", "get_weather", '"广州"}'),
                _tc(1, "call_1", "get_stock", '"600000"}'),
                _tc(2, "call_2", "get_time", '"UTC+8"}'),
            ]),
            _chunk("chunk_3", finish_reason="tool_calls", usage=_usage()),
        ])
        starts = [e[1] for e in events if e[0] == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1, 2]
        assert [s["content_block"]["id"] for s in starts] == ["call_0", "call_1", "call_2"]
        stops = [e[1] for e in events if e[0] == "content_block_stop"]
        assert [s["index"] for s in stops] == [0, 1, 2]
        assert events[-2][1]["delta"]["stop_reason"] == "tool_use"

    def test_usage_on_chunk_without_choices_carried_to_message_delta(self):
        # OpenAI 标准：usage 挂在"空 choices"的尾 chunk；
        # deepseek 则挂在带 choices 的尾 chunk——两者都要搬运到 message_delta
        usage_chunk = SimpleNamespace(id="chunk_2", choices=[], usage=_usage(100, 50))
        events = _collect([
            _chunk("chunk_1", content="hi"),
            usage_chunk,
            _chunk("chunk_3", finish_reason="stop"),
        ])
        assert events[-2][1]["usage"] == {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0}

    def test_vision_prefix_stream_emits_thinking_block_first(self):
        events = _collect([
            _chunk("chunk_1", content="你好"),
            _chunk("chunk_2", content="，世界", finish_reason="stop", usage=_usage()),
        ], vision_prefix="VLM 描述")
        # 事件序列：thinking 块 = start → thinking_delta → signature_delta → stop
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",  # thinking_delta
            "content_block_delta",  # signature_delta
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        # thinking 块 index 0，signature 走 signature_delta 事件
        assert events[1][1]["index"] == 0
        assert events[1][1]["content_block"]["type"] == "thinking"
        assert events[1][1]["content_block"]["signature"] == ""  # start 时 signature 留空
        assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "VLM 描述"}
        assert events[3][1]["delta"]["type"] == "signature_delta"  # 关键：signature_delta 事件
        assert events[3][1]["delta"]["signature"]
        assert events[4][1]["index"] == 0
        # 后续 text 块 index 从 1 开始
        assert events[5][1]["index"] == 1
        assert events[5][1]["content_block"]["type"] == "text"
        assert events[6][1]["delta"]["text"] == "你好"
        assert events[7][1]["delta"]["text"] == "，世界"
        assert events[8][1]["index"] == 1
        assert events[9][1]["delta"]["stop_reason"] == "end_turn"

    def test_vision_prefix_stream_with_tool_use_indices(self):
        events = _collect([
            _chunk("chunk_1", tool_calls=[_tc(0, "call_0", "get_weather", '{"city": "广州"}')]),
            _chunk("chunk_2", finish_reason="tool_calls", usage=_usage()),
        ], vision_prefix="VLM 描述")
        starts = [e[1] for e in events if e[0] == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1]
        assert [s["content_block"]["type"] for s in starts] == ["thinking", "tool_use"]
        stops = [e[1] for e in events if e[0] == "content_block_stop"]
        assert [s["index"] for s in stops] == [0, 1]
        assert events[-2][1]["delta"]["stop_reason"] == "tool_use"

    def test_vision_prefix_longer_than_2000_chars_sharded(self):
        prefix = "描述" * 1200  # 2400 字符，跨 2 个分片
        events = _collect([
            _chunk("chunk_1", content="ok", finish_reason="stop", usage=_usage()),
        ], vision_prefix=prefix)
        deltas = [
            e[1]["delta"]
            for e in events
            if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "thinking_delta"
        ]
        assert len(deltas) == 2
        assert all(len(d["thinking"]) <= 2000 for d in deltas)
        assert "".join(d["thinking"] for d in deltas) == prefix
        # 全部 thinking delta 的 index 都是 0
        assert all(
            e[1]["index"] == 0
            for e in events
            if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "thinking_delta"
        )

    def test_no_vision_prefix_stream_has_no_thinking_events(self):
        events = _collect([
            _chunk("chunk_1", content="hi", finish_reason="stop", usage=_usage()),
        ])
        assert [e[0] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert not any(
            e[1].get("content_block", {}).get("type") == "thinking"
            for e in events if e[0] == "content_block_start"
        )
        assert not any(
            e[1].get("delta", {}).get("type") == "thinking_delta"
            for e in events if e[0] == "content_block_delta"
        )
        # index 仍从 0 开始（与改动前一致）
        assert events[1][1]["index"] == 0
