"""merger.py 单元测试：块结构（list[dict]）、块内 --- 分段、空段省略、顺序保真。"""
from src import merger


class TestMergeImageBlock:
    def test_all_three_sections_separated_by_dashes(self):
        out = merger.merge_image_block("整体描述", "重点描述", "判断回答")
        assert out == "整体描述\n---\n重点描述\n---\n判断回答"

    def test_focus_none_omitted(self):
        out = merger.merge_image_block("整体描述", None, "判断回答")
        assert out == "整体描述\n---\n判断回答"

    def test_judgment_empty_string_omitted(self):
        out = merger.merge_image_block("整体描述", "重点描述", "")
        assert out == "整体描述\n---\n重点描述"

    def test_whitespace_sections_stripped_and_omitted(self):
        out = merger.merge_image_block("  整体描述  ", "  ", "   ")
        assert out == "整体描述"

    def test_all_empty_returns_empty_string(self):
        assert merger.merge_image_block("", None, None) == ""


class TestMergeMultiImage:
    def test_returns_list_of_text_blocks(self):
        out = merger.merge_multi_image(
            [{"overall": "整体", "focus": "重点", "judgment": "判断"}]
        )
        assert isinstance(out, list)
        assert out == [{"type": "text", "text": "整体\n---\n重点\n---\n判断"}]

    def test_single_image_one_block(self):
        out = merger.merge_multi_image(
            [{"overall": "整体", "focus": None, "judgment": None}]
        )
        assert len(out) == 1
        assert out[0] == {"type": "text", "text": "整体"}

    def test_two_images_two_blocks_in_sending_order(self):
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点", "judgment": "图1判断"},
            {"overall": "图2描述", "focus": None, "judgment": "图2判断"},
        ])
        assert len(out) == 2
        assert out[0]["text"] == "图1描述\n---\n图1重点\n---\n图1判断"
        assert out[1]["text"] == "图2描述\n---\n图2判断"

    def test_missing_judgment_key_omitted(self):
        # 旧缓存条目没有 judgment 键：不崩，该图判断段省略
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点"},
            {"overall": "图2描述", "focus": None, "judgment": "图2判断"},
        ])
        assert out[0]["text"] == "图1描述\n---\n图1重点"
        assert out[1]["text"] == "图2描述\n---\n图2判断"

    def test_all_empty_sections_emit_empty_block(self):
        out = merger.merge_multi_image([{"overall": "", "focus": "", "judgment": ""}])
        assert out == [{"type": "text", "text": ""}]
