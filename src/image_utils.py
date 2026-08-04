"""Image parsing (URL / base64 data URL), download, compression, base64 re-encode."""
import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

MAX_LONG_EDGE = 1024
JPEG_QUALITY = 85
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0)


class ImageParseError(Exception):
    """Raised when an image cannot be fetched or decoded."""


def _parse_data_url(url: str) -> bytes:
    header, sep, payload = url.partition(",")
    if not sep or not header.startswith("data:image/"):
        raise ImageParseError("unsupported data url (expected data:image/...;base64,...)")
    try:
        if ";base64" in header:
            return base64.b64decode(payload)
        return payload.encode("utf-8")
    except Exception as exc:
        raise ImageParseError(f"invalid base64 data url: {exc}") from exc


async def _download(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPError as exc:
        raise ImageParseError(f"failed to download image: {exc}") from exc


def _process_image(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ImageParseError(f"not a decodable image: {exc}") from exc

    fmt = (img.format or "JPEG").upper()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")

    width, height = img.size
    long_edge = max(width, height)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        img = img.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    if fmt == "PNG" and img.mode == "RGBA":
        img.save(out, format="PNG")
        mime = "image/png"
    else:
        img.convert("RGB").save(out, format="JPEG", quality=JPEG_QUALITY)
        mime = "image/jpeg"

    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def prepare_image(url: str) -> str:
    """Fetch and compress an image to a base64 data URL (long edge <= 1024px)."""
    if url.startswith("data:"):
        raw = _parse_data_url(url)
    elif url.startswith(("http://", "https://")):
        raw = await _download(url)
    else:
        raise ImageParseError("unsupported image url scheme (expected http(s) or data URL)")
    return await asyncio.to_thread(_process_image, raw)
