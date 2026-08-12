"""config.py 对外模型名 -> 上游模型名映射单元测试（无网络）。"""
from src import config


class TestResolveUpstreamModel:
    def test_flash_public_maps_to_flash_upstream(self):
        assert config.resolve_upstream_model(config.PUBLIC_MODEL_NAME) == config.DEEPSEEK_MODEL
        assert config.resolve_upstream_model("deepseek-v4-flash-vl") == "deepseek-v4-flash"

    def test_pro_public_maps_to_pro_upstream(self):
        assert (
            config.resolve_upstream_model(config.PUBLIC_PRO_MODEL_NAME)
            == config.DEEPSEEK_PRO_MODEL
        )
        assert config.resolve_upstream_model("deepseek-v4-pro-vl") == "deepseek-v4-pro"

    def test_unknown_model_falls_back_to_flash(self):
        assert config.resolve_upstream_model("some-other-model") == config.DEEPSEEK_MODEL

    def test_none_falls_back_to_flash(self):
        assert config.resolve_upstream_model(None) == config.DEEPSEEK_MODEL

    def test_empty_string_falls_back_to_flash(self):
        assert config.resolve_upstream_model("") == config.DEEPSEEK_MODEL
