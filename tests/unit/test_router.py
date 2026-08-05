"""router.py 纯函数单元测试（无网络、无真实 key、不启动服务）。

覆盖全部真实修复过的坑：
- _extract_current_images：WorkBuddy 双 user 分离形态、历史轮次图片隔离、
  同轮去重、10 张截断、纯图 tool 消息回退
- _normalize_tool_pairing：tool 消息被 user 文本隔开的重排（Claude Code 形态）
- _strip_message_images：纯图 tool 消息回填缓存整体描述 / 占位文本（曾回填空串）
- _desc_cache_key：同图不同问题必须不同缓存键（焦点串味修复）
"""
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

    def test_text_image_mixed_flattened_to_string(self):
        msg = {"role": "user", "content": [
            _text_part("看图 "), _img_part(self.URL), _text_part(" 回答"),
        ]}
        out = _strip_message_images(msg)
        assert out["content"] == "看图  回答"

    def test_pure_image_user_dropped(self):
        out = _strip_message_images({"role": "user", "content": [_img_part(self.URL)]})
        assert out is None


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
    def test_single_image_no_numbering(self):
        out = merger.merge_multi_image([{"overall": "画面描述", "focus": "重点描述"}], "问题")
        assert out == (
            "【图片·整体】\n画面描述\n\n【图片·重点】\n重点描述\n\n【用户问题】\n问题"
        )

    def test_multi_image_numbered_by_sending_order(self):
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点"},
            {"overall": "图2描述", "focus": None},
        ], "问题")
        # 多图编号在标签后：【图片·整体】1（单图无编号）
        assert "【图片·整体】1\n图1描述" in out
        assert "【图片·重点】1\n图1重点" in out
        assert "【图片·整体】2\n图2描述" in out
        assert "【图片·重点】2" not in out
        assert out.endswith("【用户问题】\n问题")

    def test_focus_none_omitted(self):
        out = merger.merge_multi_image([{"overall": "描述", "focus": None}], "问题")
        assert "【图片·重点】" not in out
        assert "【用户问题】\n问题" in out
