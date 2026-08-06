"""merger.py 单元测试：judgment 判断段出现/省略、单图无编号/多图编号。"""
from src import merger


class TestMergeImageInfo:
    def test_judgment_section_present(self):
        out = merger.merge_image_info("整体描述", "重点描述", "问题", "判断回答")
        assert out == (
            "【图片·整体】\n整体描述\n\n"
            "【图片·重点】\n重点描述\n\n"
            "【图片·判断】\n判断回答\n\n"
            "【用户问题】\n问题"
        )

    def test_judgment_none_omitted(self):
        out = merger.merge_image_info("整体描述", "重点描述", "问题")
        assert "【图片·判断】" not in out
        assert "【图片·重点】\n重点描述" in out
        assert out.endswith("【用户问题】\n问题")

    def test_judgment_empty_string_omitted(self):
        out = merger.merge_image_info("整体描述", None, "问题", "")
        assert "【图片·判断】" not in out


class TestMergeMultiImage:
    def test_single_image_judgment_no_numbering(self):
        out = merger.merge_multi_image(
            [{"overall": "整体", "focus": "重点", "judgment": "判断"}], "问题"
        )
        assert "【图片·判断】\n判断" in out
        assert "【图片·判断】1" not in out

    def test_multi_image_judgment_numbered_by_sending_order(self):
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点", "judgment": "图1判断"},
            {"overall": "图2描述", "focus": None, "judgment": "图2判断"},
        ], "问题")
        assert "【图片·判断】1\n图1判断" in out
        assert "【图片·判断】2\n图2判断" in out
        assert out.endswith("【用户问题】\n问题")

    def test_multi_image_missing_judgment_key_omitted(self):
        # 旧缓存条目没有 judgment 键：不崩，该图判断段省略
        out = merger.merge_multi_image([
            {"overall": "图1描述", "focus": "图1重点"},
            {"overall": "图2描述", "focus": None, "judgment": "图2判断"},
        ], "问题")
        assert "【图片·判断】1" not in out
        assert "【图片·判断】2\n图2判断" in out

    def test_judgment_empty_omitted_in_multi(self):
        out = merger.merge_multi_image([
            {"overall": "图1", "focus": None, "judgment": ""},
        ], "问题")
        assert "【图片·判断】" not in out

    def test_layout_order(self):
        out = merger.merge_multi_image(
            [{"overall": "整体", "focus": "重点", "judgment": "判断"}], "问题"
        )
        idxs = [
            out.index("【图片·整体】"),
            out.index("【图片·重点】"),
            out.index("【图片·判断】"),
            out.index("【用户问题】"),
        ]
        assert idxs == sorted(idxs)
