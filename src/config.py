"""Environment configuration, loaded from the project root .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_VLM_API_KEY = os.getenv("DEEPSEEK_VLM_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
# 对外声称的模型名（客户端看到的 model id / 响应回显），可不同于上游模型名。
PUBLIC_MODEL_NAME = os.getenv("PUBLIC_MODEL_NAME", "deepseek-v4-flash-vl")
PUBLIC_PRO_MODEL_NAME = os.getenv("PUBLIC_PRO_MODEL_NAME", "deepseek-v4-pro-vl")
PORT = int(os.getenv("PORT", "8000"))

# 对外模型名 -> 上游模型名
MODEL_MAP = {
    PUBLIC_MODEL_NAME: DEEPSEEK_MODEL,          # deepseek-v4-flash-vl -> deepseek-v4-flash
    PUBLIC_PRO_MODEL_NAME: DEEPSEEK_PRO_MODEL,  # deepseek-v4-pro-vl -> deepseek-v4-pro
}


def resolve_upstream_model(model: str | None) -> str:
    """请求的对外模型名 -> 上游 deepseek 模型名；未知/空 -> 默认 flash 上游。"""
    if model and model in MODEL_MAP:
        return MODEL_MAP[model]
    return DEEPSEEK_MODEL
