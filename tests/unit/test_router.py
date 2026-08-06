"""router.py 纯函数单元测试（无网络、无真实 key、不启动服务）。

覆盖全部真实修复过的坑：
- _extract_current_images：WorkBuddy 双 user 分离形态、历史轮次图片隔离、
  同轮去重、10 张截断、纯图 tool 消息回退
- _normalize_tool_pairing：tool 消息被 user 文本隔开的重排（Claude Code 形态）
- _strip_message_images：当前轮 [图片 N] / 历史 [历史图片] 占位符保序；
  纯图 tool 消息回填缓存整体描述 / 占位文本（曾回填空串）
- _desc_cache_key：同图不同问题必须不同缓存键（焦点串味修复）
- _forward vision_prefix：VLM 描述注入 assistant reasoning_content 前缀
- 组装布局：历史 → user(问题+[图片 N]) → user(merged 块数组) → user(LLM_SYSTEM)
"""
import asyncio
import json

import pytest

from src import image_utils, merger, router
from src.router import (
    _cache_desc,
    _desc_cache_key,
    _ensure_reasoning_content,
    _extract_current_images,
    _find_cached_overall,
    _normalize_tool_pairing,
    _parse_content,
    _parse_stream,
    _strip_message_images,
)

# _DESC_CACHE 是模块级全局，测试间必须清空
@pytest.fixture(autouse=True)
def clear_desc_cache():
    router._DESC_CACHE.clear()
    yield
    router._DESC_CACHE.clear()


def _img_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _text_part(t):
    return {"type": "text", "text": t}


def _user(content):
    return {"role": "user", "content": content}


def _assistant_with_tools(*ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "f", "arguments": "{}"}}
            for tid in ids
        ],
    }


def _tool_msg(tid, text="结果"):
    return {"role": "tool", "tool_call_id": tid, "content": text}


# ---------- _parse_content ----------

class TestParseContent:
    def test_string(self):
        assert _parse_content("你好") == ("你好", [])

    def test_list_of_text_parts(self):
        content = [_text_part("a"), _text_part("b")]
        assert _parse_content(content) == ("ab", [])

    def test_image_url_object_format(self):
        assert _parse_content([_img_part("http://img/x.png")]) == ("", ["http://img/x.png"])

    def test_image_url_string_format(self):
        # WorkBuddy 可能把 image_url 直接发成字符串
        assert _parse_content([{"type": "image_url", "image_url": "http://img/x.png"}]) == (
            "",
            ["http://img/x.png"],
        )

    def test_no_type_field_compat_branch(self):
        assert _parse_content([{"text": "旧格式文本"}]) == ("旧格式文本", [])
        assert _parse_content([{"image_url": "http://img/x.png"}]) == ("", ["http://img/x.png"])

    def test_invalid_content_raises(self):
        with pytest.raises(router.ClientRequestError):
            _parse_content(123)


# ---------- _extract_current_images ----------

class TestExtractCurrentImages:
    def test_single_user_with_image(self):
        msgs = [_user([_text_part("看这张图"), _img_part("http://img/a.png")])]
        assert _extract_current_images(msgs) == ["http://img/a.png"]

    def test_dual_user_separated_image_and_text(self):
        # WorkBuddy 形态：图片消息与文本消息分成两条 user——跳过无图 user 继续倒序扫描
        msgs = [
            _user([_img_part("http://img/a.png")]),
            _user("这个按钮为什么点不了"),
        ]
        assert _extract_current_images(msgs) == ["http://img/a.png"]

    def test_three_users_three_images_original_order(self):
        msgs = [
            _user([_img_part("http://img/1.png")]),
            _user([_img_part("http://img/2.png")]),
            _user([_img_part("http://img/3.png")]),
        ]
        assert _extract_current_images(msgs) == [
            "http://img/1.png", "http://img/2.png", "http://img/3.png",
        ]

    def test_one_user_three_images(self):
        msgs = [_user([
            _img_part("http://img/1.png"),
            _img_part("http://img/2.png"),
            _img_part("http://img/3.png"),
        ])]
        assert _extract_current_images(msgs) == [
            "http://img/1.png", "http://img/2.png", "http://img/3.png",
        ]

    def test_history_images_not_collected(self):
        # 历史轮次图片被 assistant 隔断 → 不误收
        msgs = [
            _user([_img_part("http://img/hist.png")]),
            {"role": "assistant", "content": "历史回答"},
            _user([_text_part("新问题"), _img_part("http://img/cur.png")]),
        ]
        assert _extract_current_images(msgs) == ["http://img/cur.png"]

    def test_duplicate_image_same_turn_deduped(self):
        msgs = [_user([_img_part("http://img/dup.png"), _img_part("http://img/dup.png")])]
        assert _extract_current_images(msgs) == ["http://img/dup.png"]

    def test_more_than_10_truncated(self):
        urls = [f"http://img/{i}.png" for i in range(12)]
        msgs = [_user([_img_part(u) for u in urls])]
        out = _extract_current_images(msgs)
        assert len(out) == 10
        assert out == urls[2:]  # 保留原始顺序的最近 10 张

    def test_fallback_to_last_tool_message(self):
        # agent 本轮回调 Read 工具：最后一条是纯图 tool 消息
        msgs = [
            _user("读一下图片"),
            {"role": "tool", "tool_call_id": "call_1", "content": [_img_part("http://img/read.png")]},
        ]
        assert _extract_current_images(msgs) == ["http://img/read.png"]


# ---------- _normalize_tool_pairing ----------

class TestNormalizeToolPairing:
    def test_normal_pairing_keeps_order(self):
        msgs = [_assistant_with_tools("call_1"), _tool_msg("call_1")]
        out = _normalize_tool_pairing(msgs)
        assert [m["role"] for m in out] == ["assistant", "tool"]
        assert out[1]["tool_call_id"] == "call_1"

    def test_tool_separated_by_user_text_reordered(self):
        # 已修复 bug 回归：Claude Code 在 tool_use 与 tool_result 之间插入
        # user 文本——tool 消息必须重排到对应 assistant 之后
        msgs = [_assistant_with_tools("call_1"), _user("继续"), _tool_msg("call_1")]
        out = _normalize_tool_pairing(msgs)
        assert [m["role"] for m in out] == ["assistant", "tool", "user"]
        assert out[1]["tool_call_id"] == "call_1"
        assert out[2]["content"] == "继续"

    def test_duplicate_tool_call_id_reuse_does_not_crash(self):
        # 跨轮复用同一 tool_call_id：不崩，tool 消息只挂第一次出现处
        msgs = [
            _assistant_with_tools("call_1"), _tool_msg("call_1"),
            _assistant_with_tools("call_1"), _tool_msg("call_1"),
        ]
        out = _normalize_tool_pairing(msgs)
        assert [m["role"] for m in out] == ["assistant", "tool", "tool", "assistant"]

    def test_orphan_tool_message_appended_at_end(self):
        # 孤儿 tool 消息（无匹配 assistant）排到末尾，不崩
        msgs = [_user("问题"), _tool_msg("call_9")]
        out = _normalize_tool_pairing(msgs)
        assert [m["role"] for m in out] == ["user", "tool"]
        assert out[-1]["tool_call_id"] == "call_9"


# ---------- _ensure_reasoning_content ----------

class TestEnsureReasoningContent:
    def test_assistant_with_tool_calls_padded_with_empty_string(self):
        msgs = [_assistant_with_tools("call_1")]
        out = _ensure_reasoning_content(msgs)
        assert out[0]["reasoning_content"] == ""

    def test_existing_reasoning_content_not_overwritten(self):
        msgs = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
            "reasoning_content": "思考中",
        }]
        out = _ensure_reasoning_content(msgs)
        assert out[0]["reasoning_content"] == "思考中"

    def test_normal_messages_untouched(self):
        msgs = [_user("hi"), {"role": "assistant", "content": "hi"}]
        out = _ensure_reasoning_content(msgs)
        assert out == msgs


class TestCurrentQuestionText:
    """_current_question_text：VLM-3 的输入只取当前问题，不扫历史。"""

    def test_returns_current_user_text_only(self):
        msgs = [
            _user("历史问题一"),
            {"role": "assistant", "content": "历史回答"},
            _user("当前问题"),
        ]
        assert router._current_question_text(msgs) == "当前问题"

    def test_multimodal_current_message_text_extracted(self):
        msgs = [
            _user("历史问题"),
            {"role": "assistant", "content": "历史回答"},
            _user([_text_part("图里是谁"), _img_part("http://img/a.png")]),
        ]
        assert router._current_question_text(msgs) == "图里是谁"

    def test_pure_image_message_returns_empty(self):
        msgs = [_user("历史问题"), {"role": "assistant", "content": "历史回答"}, _user([_img_part("http://img/a.png")])]
        assert router._current_question_text(msgs) == ""

    def test_last_user_idx_explicit(self):
        msgs = [_user("历史"), _user("当前")]
        assert router._current_question_text(msgs, last_user_idx=1) == "当前"


# ---------- _strip_message_images ----------

class TestStripMessageImages:
    URL = "http://img/read.png"

    def _pure_image_tool_msg(self):
        return {"role": "tool", "tool_call_id": "call_1", "content": [_img_part(self.URL)]}

    def test_pure_image_tool_backfills_cached_overall(self):
        _cache_desc(_desc_cache_key(image_utils.image_hash(self.URL), "问题"), "整体描述A", None)
        out = _strip_message_images(self._pure_image_tool_msg())
        assert out["content"] == "整体描述A"

    def test_pure_image_tool_cache_miss_placeholder_not_empty(self):
        # 已修复 bug 回归：曾回填空串，deepseek 误判"read_file 返回空"；
        # 必须回填非空占位文本
        out = _strip_message_images(self._pure_image_tool_msg())
        assert isinstance(out["content"], str)
        assert out["content"]
        assert out["content"].startswith("【图片】此消息包含一张图片")

    def test_text_image_mixed_kept_with_current_placeholder(self):
        # [text1, image, text2] → [text1, [图片 1], text2] 保序（不再压平）
        msg = {"role": "user", "content": [
            _text_part("看图 "), _img_part(self.URL), _text_part(" 回答"),
        ]}
        out = _strip_message_images(msg, current_image_urls=[self.URL])
        assert out["content"] == [
            _text_part("看图 "),
            _text_part("[图片 1]"),
            _text_part(" 回答"),
        ]

    def test_mixed_history_images_become_history_placeholder(self):
        msg = {"role": "user", "content": [
            _text_part("看图 "), _img_part(self.URL), _text_part(" 回答"),
        ]}
        out = _strip_message_images(msg)
        assert out["content"] == [
            _text_part("看图 "),
            _text_part("[历史图片]"),
            _text_part(" 回答"),
        ]

    def test_pure_image_user_kept_with_current_placeholder(self):
        # 纯图 user 消息不再 drop → [图片 1]
        out = _strip_message_images(
            {"role": "user", "content": [_img_part(self.URL)]},
            current_image_urls=[self.URL],
        )
        assert out["content"] == [_text_part("[图片 1]")]

    def test_pure_image_user_history_placeholder(self):
        out = _strip_message_images({"role": "user", "content": [_img_part(self.URL)]})
        assert out["content"] == [_text_part("[历史图片]")]

    def test_assistant_image_history_placeholder(self):
        msg = {"role": "assistant", "content": [
            _text_part("收到 "), _img_part(self.URL),
        ]}
        out = _strip_message_images(msg)
        assert out["content"] == [_text_part("收到 "), _text_part("[历史图片]")]

    def test_current_placeholders_numbered_by_url_index(self):
        # N = url 在 current_image_urls 中的下标+1，与 merged 块一一对应
        urls = ["http://img/1.png", "http://img/2.png"]
        msg = {"role": "user", "content": [_img_part(urls[1]), _img_part(urls[0])]}
        out = _strip_message_images(msg, current_image_urls=urls)
        assert out["content"] == [_text_part("[图片 2]"), _text_part("[图片 1]")]

    def test_current_placeholder_matches_extract_current_images_order(self):
        # 真实流程：cur_images 来自 _extract_current_images（原始发送顺序）
        msgs = [_user([_img_part("http://img/a.png"), _img_part("http://img/b.png")])]
        cur = _extract_current_images(msgs)
        out = _strip_message_images(msgs[0], current_image_urls=cur)
        assert out["content"] == [_text_part("[图片 1]"), _text_part("[图片 2]")]

    def test_url_not_in_current_list_falls_back_to_history_placeholder(self):
        out = _strip_message_images(
            {"role": "user", "content": [_img_part("http://img/other.png")]},
            current_image_urls=["http://img/a.png"],
        )
        assert out["content"] == [_text_part("[历史图片]")]

    def test_text_only_array_still_flattened(self):
        out = _strip_message_images({"role": "user", "content": [_text_part("a"), _text_part("b")]})
        assert out["content"] == "ab"


# ---------- 描述缓存键 ----------

class TestDescCache:
    def test_same_image_different_question_different_keys(self):
        # 焦点串味修复回归：VLM-2 focus 是问题驱动的，
        # 同一张图不同问题必须命中不同缓存条目
        k1 = _desc_cache_key("hashA", "这个按钮为什么点不了")
        k2 = _desc_cache_key("hashA", "图片里有哪些文字")
        assert k1 != k2
        assert k1.startswith("hashA|") and k2.startswith("hashA|")

    def test_find_cached_overall_ignores_question_fingerprint(self):
        # 纯图 tool_result 回填没有问题可用，_find_cached_overall
        # 必须忽略问题指纹找到整体描述
        _cache_desc(_desc_cache_key("hashB", "问题1"), "整体1", "重点1")
        assert _find_cached_overall("hashB") == "整体1"

    def test_find_cached_overall_returns_most_recent_entry(self):
        _cache_desc(_desc_cache_key("hashC", "问题1"), "整体1", None)
        _cache_desc(_desc_cache_key("hashC", "问题2"), "整体2", None)
        assert _find_cached_overall("hashC") == "整体2"

    def test_find_cached_overall_miss_returns_none(self):
        assert _find_cached_overall("hashX") is None


# ---------- _parse_stream ----------

class TestParseStream:
    @pytest.mark.parametrize("value", ["false", "0", "", "False", "FALSE"])
    def test_falsy_strings(self, value):
        assert _parse_stream(value) is False

    def test_falsy_bool(self):
        assert _parse_stream(False) is False

    @pytest.mark.parametrize("value", ["true", "1", "True", "TRUE"])
    def test_truthy_strings(self, value):
        assert _parse_stream(value) is True

    def test_truthy_bool(self):
        assert _parse_stream(True) is True


# ---------- merger.merge_multi_image ----------

class TestMergeMultiImage:
    def test_single_image_one_block(self):
        out = merger.merge_multi_image([{"overall": "画面描述", "focus": "重点描述"}])
        assert out == [{"type": "text", "text": "画面描述\n---\n重点描述"}]

    def test_multi_image_blocks_in_sending_order(self):
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点"},
            {"overall": "图2描述", "focus": None},
        ])
        assert len(out) == 2
        assert out[0]["text"] == "图1描述\n---\n图1重点"
        assert out[1]["text"] == "图2描述"

    def test_focus_none_omitted(self):
        out = merger.merge_multi_image([{"overall": "描述", "focus": None}])
        assert out == [{"type": "text", "text": "描述"}]


# ---------- _forward vision_prefix ----------

class TestForwardVisionPrefix:
    """_forward 的 vision_prefix 注入（monkeypatch _llm，无网络）。"""

    MSGS = [_user("看图回答")]
    MODEL = "deepseek-v4-flash-vl"

    @staticmethod
    def _reply(content="回答正文", reasoning=None):
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }

    def _patch_complete(self, monkeypatch, data):
        async def fake_complete(messages, body):
            return data

        monkeypatch.setattr(router._llm, "complete", fake_complete)

    def test_non_streaming_reasoning_content_prefixed_content_unchanged(self, monkeypatch):
        # 有图轮：message.reasoning_content 以 vision_prefix 开头（无旧值时
        # 等于前缀），content 回答正文不变
        self._patch_complete(monkeypatch, self._reply())
        resp = asyncio.run(
            router._forward(self.MSGS, {}, self.MODEL, False, vision_prefix="VLM描述")
        )
        msg = json.loads(resp.body)["choices"][0]["message"]
        assert msg["reasoning_content"] == "VLM描述"
        assert msg["content"] == "回答正文"

    def test_non_streaming_existing_reasoning_concatenated(self, monkeypatch):
        # deepseek thinking 模式已有 reasoning_content：VLM 描述在前，
        # 旧思考内容接在后面（\n\n 分隔）
        self._patch_complete(monkeypatch, self._reply(reasoning="deepseek思考"))
        resp = asyncio.run(
            router._forward(self.MSGS, {}, self.MODEL, False, vision_prefix="VLM描述")
        )
        msg = json.loads(resp.body)["choices"][0]["message"]
        assert msg["reasoning_content"] == "VLM描述\n\ndeepseek思考"
        assert msg["content"] == "回答正文"

    def test_non_streaming_without_prefix_unchanged(self, monkeypatch):
        # 无图轮：vision_prefix 为 None，response 原样透传
        self._patch_complete(monkeypatch, self._reply(reasoning="deepseek思考"))
        resp = asyncio.run(router._forward(self.MSGS, {}, self.MODEL, False))
        msg = json.loads(resp.body)["choices"][0]["message"]
        assert msg["reasoning_content"] == "deepseek思考"
        assert msg["content"] == "回答正文"

    def test_streaming_first_chunk_is_vision_prefix(self, monkeypatch):
        # 有图轮流式：首个 SSE chunk 的 delta 只有 reasoning_content ==
        # vision_prefix（无 content），其后才透传 deepseek 的流
        async def fake_stream(messages, body, model):
            async def gen():
                yield (
                    "data: "
                    + json.dumps({
                        "id": "chatcmpl_ds",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": "你好"},
                            "finish_reason": None,
                        }],
                    })
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return gen()

        monkeypatch.setattr(router._llm, "stream", fake_stream)
        resp = asyncio.run(
            router._forward(self.MSGS, {}, self.MODEL, True, vision_prefix="VLM描述")
        )

        async def collect():
            return [line async for line in resp.body_iterator]

        lines = asyncio.run(collect())
        first = json.loads(lines[0][len("data: "):].strip())
        assert first["choices"][0]["delta"]["reasoning_content"] == "VLM描述"
        assert first["choices"][0]["delta"].get("content") is None
        assert first["choices"][0]["finish_reason"] is None
        assert first["model"] == self.MODEL
        # deepseek 原始 chunk 原样透传
        second = json.loads(lines[1][len("data: "):].strip())
        assert second["choices"][0]["delta"]["content"] == "你好"
        assert lines[-1] == "data: [DONE]\n\n"


# ---------- VLM-3 三路并发 ----------

class TestVlmPairThreeWay:
    """有图轮 vlm_pair 三路并发（monkeypatch _vlm / prepare_image / _llm.complete，无网络）。"""

    URL = "http://img/v3.png"

    @staticmethod
    def _reply():
        return {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "回答正文"},
                "finish_reason": "stop",
            }],
        }

    def _patch_vlm(self, monkeypatch, calls):
        async def fake_overall(system_prompt, data_url):
            calls.append("overall")
            return "整体描述"

        async def fake_focus(system_prompt, data_url, question):
            calls.append("focus")
            return "重点描述"

        async def fake_judgment(system_prompt, data_url, question):
            calls.append("judgment")
            return "判断描述"

        async def fake_prepare(url):
            return "data:image/png;base64,AAAA"

        async def fake_complete(messages, body):
            return self._reply()

        monkeypatch.setattr(router._vlm, "describe_overall", fake_overall)
        monkeypatch.setattr(router._vlm, "describe_focus", fake_focus)
        monkeypatch.setattr(router._vlm, "describe_judgment", fake_judgment)
        monkeypatch.setattr(image_utils, "prepare_image", fake_prepare)
        monkeypatch.setattr(router._llm, "complete", fake_complete)

    def test_image_with_question_runs_three_vlms(self, monkeypatch):
        calls = []
        self._patch_vlm(monkeypatch, calls)
        msgs = [_user([_text_part("这是谁"), _img_part(self.URL)])]
        resp = asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "deepseek-v4-flash-vl", "stream": False}
            )
        )
        assert set(calls) == {"overall", "focus", "judgment"}
        # judgment 进入合并块文本（vision_prefix → reasoning_content）
        msg = json.loads(resp.body)["choices"][0]["message"]
        assert "整体描述\n---\n重点描述\n---\n判断描述" in msg["reasoning_content"]
        # 缓存条目含 judgment
        assert len(router._DESC_CACHE) == 1
        assert next(iter(router._DESC_CACHE.values()))["judgment"] == "判断描述"

    def test_pure_image_without_question_skips_focus(self, monkeypatch):
        # run_vlm2 = len(cur_text.strip()) > 1：纯图无文本 → focus 不跑，
        # judgment 仍跑（判断不依赖问题文本，有图就判断）
        calls = []
        self._patch_vlm(monkeypatch, calls)
        msgs = [_user([_img_part(self.URL)])]
        asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        assert "focus" not in calls
        assert set(calls) == {"overall", "judgment"}
        assert len(router._DESC_CACHE) == 1
        assert next(iter(router._DESC_CACHE.values()))["judgment"] == "判断描述"

    def test_judgment_gets_current_question_not_history(self, monkeypatch):
        # VLM-3 的输入必须只含当前问题，不能带历史对话上下文（防止判断被
        # 前序轮次污染）。构造：历史有 user/assistant 对话 + 当前轮带图问题。
        judgment_questions = []

        async def fake_judgment(system_prompt, data_url, question):
            judgment_questions.append(question)
            return "判断描述"

        async def fake_overall(system_prompt, data_url):
            return "整体描述"

        async def fake_focus(system_prompt, data_url, question):
            return "重点描述"

        async def fake_prepare(url):
            return "data:image/png;base64,AAAA"

        async def fake_complete(messages, body):
            return self._reply()

        monkeypatch.setattr(router._vlm, "describe_overall", fake_overall)
        monkeypatch.setattr(router._vlm, "describe_focus", fake_focus)
        monkeypatch.setattr(router._vlm, "describe_judgment", fake_judgment)
        monkeypatch.setattr(image_utils, "prepare_image", fake_prepare)
        monkeypatch.setattr(router._llm, "complete", fake_complete)

        msgs = [
            _user("之前聊过什么"),
            {"role": "assistant", "content": "之前回答过"},
            _user([_text_part("这是谁"), _img_part(self.URL)]),
        ]
        asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        assert len(judgment_questions) == 1
        assert judgment_questions[0] == "这是谁"
        assert "之前聊过什么" not in judgment_questions[0]
        assert "之前回答过" not in judgment_questions[0]


# ---------- 旧缓存条目兼容（无 judgment） ----------

class TestCacheBackwardCompat:
    """旧缓存条目没有 judgment：读取与合并不崩，判断段省略。"""

    URL = "http://img/old.png"

    def test_history_merge_old_entry_without_judgment(self):
        msgs = [
            _user([_img_part(self.URL)]),
            {"role": "assistant", "content": "看过了"},
            _user("那张图里是谁？"),
        ]
        cur_text = router._pick_focus_text(msgs)
        key = _desc_cache_key(image_utils.image_hash(self.URL), cur_text)
        router._DESC_CACHE[key] = {"overall": "旧整体", "focus": "旧重点"}  # 旧格式
        merged = router._find_cached_history_image(msgs, cur_text)
        assert merged is not None
        assert merged == [{"type": "text", "text": "旧整体\n---\n旧重点"}]

    def test_history_merge_new_entry_with_judgment(self):
        msgs = [
            _user([_img_part(self.URL)]),
            {"role": "assistant", "content": "看过了"},
            _user("那张图里是谁？"),
        ]
        cur_text = router._pick_focus_text(msgs)
        key = _desc_cache_key(image_utils.image_hash(self.URL), cur_text)
        router._DESC_CACHE[key] = {
            "overall": "整体", "focus": "重点", "judgment": "迪丽热巴",
        }
        merged = router._find_cached_history_image(msgs, cur_text)
        assert merged == [{"type": "text", "text": "整体\n---\n重点\n---\n迪丽热巴"}]

    def test_image_turn_old_entry_cache_hit_no_crash(self, monkeypatch):
        # 有图轮命中旧格式缓存：不下载不调 VLM，合并输出无判断段
        msgs = [_user([_text_part("问题"), _img_part(self.URL)])]
        cur_text = router._pick_focus_text(msgs)
        key = _desc_cache_key(image_utils.image_hash(self.URL), cur_text)
        router._DESC_CACHE[key] = {"overall": "旧整体", "focus": "旧重点"}

        async def fake_complete(messages, body):
            return {
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "回答"},
                    "finish_reason": "stop",
                }],
            }

        monkeypatch.setattr(router._llm, "complete", fake_complete)
        # 不 patch prepare_image：缓存命中必须完全跳过下载/压缩
        resp = asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        msg = json.loads(resp.body)["choices"][0]["message"]
        assert "旧整体\n---\n旧重点" in msg["reasoning_content"]


# ---------- 组装布局 ----------

class TestAssemblyLayout:
    """组装布局：历史 → user(问题+[图片 N]) → user(merged 块数组) → user(LLM_SYSTEM)。"""

    URL1 = "http://img/l1.png"
    URL2 = "http://img/l2.png"

    @staticmethod
    def _reply():
        return {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "回答"},
                "finish_reason": "stop",
            }],
        }

    def _patch_vlm_llm(self, monkeypatch, captured):
        async def fake_overall(system_prompt, data_url):
            return "整体描述"

        async def fake_focus(system_prompt, data_url, question):
            return "重点描述"

        async def fake_judgment(system_prompt, data_url, question):
            return "判断描述"

        async def fake_prepare(url):
            return "data:image/png;base64,AAAA"

        async def fake_complete(messages, body):
            captured.append(messages)
            return self._reply()

        monkeypatch.setattr(router._vlm, "describe_overall", fake_overall)
        monkeypatch.setattr(router._vlm, "describe_focus", fake_focus)
        monkeypatch.setattr(router._vlm, "describe_judgment", fake_judgment)
        monkeypatch.setattr(image_utils, "prepare_image", fake_prepare)
        monkeypatch.setattr(router._llm, "complete", fake_complete)

    def test_image_turn_layout(self, monkeypatch):
        # [text1, image, text2] + 当前轮问题 → 保留原消息 + [图片 N]
        captured = []
        self._patch_vlm_llm(monkeypatch, captured)
        msgs = [
            _user("历史问题"),
            {"role": "assistant", "content": "历史回答"},
            _user([
                _text_part("图里是什么 "),
                _img_part(self.URL1),
                _text_part(" ？"),
                _img_part(self.URL2),
            ]),
        ]
        asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        out = captured[0]
        # 历史消息原样保留（无图）
        assert out[0] == msgs[0]
        assert out[1] == msgs[1]
        # 当前轮 user：问题文本 + [图片 N] 占位符，保序
        assert out[2]["role"] == "user"
        assert out[2]["content"] == [
            _text_part("图里是什么 "),
            _text_part("[图片 1]"),
            _text_part(" ？"),
            _text_part("[图片 2]"),
        ]
        # merged 块数组：2 块（数组下标 ↔ 图片编号），块内 --- 分隔三段
        assert out[3]["role"] == "user"
        assert out[3]["content"] == [
            {"type": "text", "text": "整体描述\n---\n重点描述\n---\n判断描述"},
            {"type": "text", "text": "整体描述\n---\n重点描述\n---\n判断描述"},
        ]
        # LLM_SYSTEM 最后
        assert out[4]["role"] == "user"
        assert out[4]["content"] == router.LLM_SYSTEM
        assert len(out) == 5

    def test_history_image_becomes_history_placeholder(self, monkeypatch):
        # 历史轮次 user 图片 → [历史图片]；当前轮 → [图片 1]
        captured = []
        self._patch_vlm_llm(monkeypatch, captured)
        msgs = [
            _user([_text_part("看图 "), _img_part("http://img/hist.png")]),
            {"role": "assistant", "content": "历史回答"},
            _user([_text_part("新问题"), _img_part(self.URL1)]),
        ]
        asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        out = captured[0]
        assert out[0]["content"] == [_text_part("看图 "), _text_part("[历史图片]")]
        assert out[2]["content"] == [_text_part("新问题"), _text_part("[图片 1]")]
        assert out[3]["content"] == [
            {"type": "text", "text": "整体描述\n---\n重点描述\n---\n判断描述"}
        ]
        assert out[4]["content"] == router.LLM_SYSTEM


class TestInjectHistoryDescription:
    """无图轮注入布局：历史 → user(问题) → user(merged 块) → user(LLM_SYSTEM)。"""

    def test_blocks_inserted_after_last_user(self):
        stripped = [
            {"role": "user", "content": "历史问题"},
            {"role": "assistant", "content": "历史回答"},
            {"role": "user", "content": "当前问题"},
        ]
        blocks = [{"type": "text", "text": "整体\n---\n重点"}]
        out = router._inject_history_description(stripped, blocks)
        assert out[:3] == stripped
        assert out[3] == {"role": "user", "content": blocks}
        assert out[4] == {"role": "user", "content": router.LLM_SYSTEM}
        assert len(out) == 5

    def test_no_image_turn_cached_history_layout(self, monkeypatch):
        # 完整无图轮流程：历史图占位 + 注入块插在当前问题之后、LLM_SYSTEM 最后
        captured = []

        async def fake_complete(messages, body):
            captured.append(messages)
            return {
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "回答"},
                    "finish_reason": "stop",
                }],
            }

        monkeypatch.setattr(router._llm, "complete", fake_complete)
        msgs = [
            _user([_img_part("http://img/hist.png")]),
            {"role": "assistant", "content": "看过了"},
            _user("那张图里是谁？"),
        ]
        cur_text = router._pick_focus_text(msgs)
        key = _desc_cache_key(image_utils.image_hash("http://img/hist.png"), cur_text)
        router._DESC_CACHE[key] = {
            "overall": "整体", "focus": "重点", "judgment": "迪丽热巴",
        }
        asyncio.run(
            router.route_chat_completions(
                {"messages": msgs, "model": "m", "stream": False}
            )
        )
        out = captured[0]
        assert out[0]["content"] == [_text_part("[历史图片]")]
        assert out[1] == msgs[1]
        assert out[2] == msgs[2]  # 当前问题保留原样
        assert out[3]["content"] == [
            {"type": "text", "text": "整体\n---\n重点\n---\n迪丽热巴"}
        ]
        assert out[4]["content"] == router.LLM_SYSTEM
        assert len(out) == 5
