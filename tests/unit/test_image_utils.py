"""image_utils.py 单元测试：SSRF 防护地址判定 + 图片内容哈希。"""
import base64

import pytest

from src.image_utils import _is_blocked_address, image_hash


class TestIsBlockedAddress:
    @pytest.mark.parametrize("ip", [
        "127.0.0.1",      # 回环
        "10.0.0.1",       # RFC1918
        "192.168.1.1",    # RFC1918
        "172.16.0.1",     # RFC1918
        "169.254.169.254",  # 链路本地（云元数据服务）
        "0.0.0.0",        # 未指定
        "::1",            # IPv6 回环
    ])
    def test_private_reserved_addresses_blocked(self, ip):
        assert _is_blocked_address(ip) is True

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "114.114.114.114"])
    def test_public_addresses_allowed(self, ip):
        assert _is_blocked_address(ip) is False


class TestImageHash:
    def test_data_url_same_content_same_hash(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64 = base64.b64encode(raw).decode()
        url = f"data:image/png;base64,{b64}"
        assert image_hash(url) == image_hash(url)

    def test_data_url_different_content_different_hash(self):
        a = image_hash("data:image/png;base64," + base64.b64encode(b"AAAA").decode())
        b = image_hash("data:image/png;base64," + base64.b64encode(b"BBBB").decode())
        assert a != b

    def test_http_url_key_is_url_string(self):
        # http URL 无法在下载前做内容哈希，键就是 url 字符串
        assert image_hash("http://img/a.png") == "url:http://img/a.png"
        assert image_hash("https://img/b.png") == "url:https://img/b.png"
