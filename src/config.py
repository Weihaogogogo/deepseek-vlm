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
# 对外声称的模型名（客户端看到的 model id / 响应回显），可不同于上游模型名。
PUBLIC_MODEL_NAME = os.getenv("PUBLIC_MODEL_NAME", "deepseek-v4-flash-vl")
PORT = int(os.getenv("PORT", "8000"))
