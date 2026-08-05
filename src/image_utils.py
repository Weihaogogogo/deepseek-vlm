"""Image parsing (URL / base64 data URL), download, compression, base64 re-encode."""
import asyncio
import base64
import hashlib
import io
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

MAX_LONG_EDGE = 1024
JPEG_QUALITY = 85
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0)
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


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


def _is_blocked_address(ip: str) -> bool:
    """True for loopback / RFC1918 / link-local / unspecified / ULA addresses."""
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if addr.version == 4:
        return (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_unspecified
        )
    return (
        addr.is_loopback or addr.is_link_local or addr.is_private or addr.is_unspecified
    )


async def _validate_url(url: str) -> None:
    """Reject URLs whose hostname resolves to private/reserved addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ImageParseError(f"unsupported url: {url[:120]}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo, parsed.hostname, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ImageParseError(f"failed to resolve image host: {parsed.hostname}") from exc
    if not infos:
        raise ImageParseError(f"failed to resolve image host: {parsed.hostname}")
    for info in infos:
        ip = info[4][0]
        if _is_blocked_address(ip):
            raise ImageParseError(
                f"blocked address (private/reserved): {ip} for host {parsed.hostname}"
            )


async def _download(url: str) -> bytes:
    """Download with SSRF guard (entry + every redirect target) and a 20MB cap."""
    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            await _validate_url(current)
            async with httpx.AsyncClient(
                timeout=DOWNLOAD_TIMEOUT, follow_redirects=False
            ) as client:
                async with client.stream("GET", current) as resp:
                    if resp.status_code in _REDIRECT_STATUSES:
                        loc = resp.headers.get("location")
                        if not loc:
                            raise ImageParseError(
                                f"redirect without location header: {current}"
                            )
                        current = str(httpx.URL(current).join(loc))
                        continue
                    resp.raise_for_status()
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_DOWNLOAD_BYTES:
                            raise ImageParseError(
                                f"image exceeds {MAX_DOWNLOAD_BYTES} byte limit"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
        raise ImageParseError(f"too many redirects downloading: {url}")
    except httpx.HTTPError as exc:
        raise ImageParseError(f"failed to download image: {exc}") from exc


def _process_image(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data))
        # JPEG: draft() uses libjpeg scale-down decoding (1/2, 1/4, 1/8) —
        # decodes only ~2x the target size instead of the full resolution.
        # Big screenshots are the slow path; this cuts decode time 4-16x.
        if img.format == "JPEG":
            try:
                img.draft("RGB", (MAX_LONG_EDGE * 2, MAX_LONG_EDGE * 2))
            except Exception:  # noqa: BLE001
                pass
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
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        if long_edge > MAX_LONG_EDGE * 2:
            # Two-step downscale: cheap BILINEAR to ~2x target, then LANCZOS
            # for the final size (LANCZOS on the full image is the slow part).
            mid = (max(1, round(width * 2 * scale)), max(1, round(height * 2 * scale)))
            img = img.resize(mid, Image.Resampling.BILINEAR)
        img = img.resize(target, Image.Resampling.LANCZOS)

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


def image_hash(url: str) -> str:
    """Content hash of an image URL (used as the VLM description cache key).

    Hashes the RAW image bytes so the same picture sent as URL or base64 maps
    to the same key. Cheap for data URLs (hash on raw bytes, before any
    download/compression).
    """
    if url.startswith("data:"):
        try:
            raw = _parse_data_url(url)
        except ImageParseError:
            return f"url:{url[:120]}"
    elif url.startswith(("http://", "https://")):
        return f"url:{url}"
    else:
        return f"raw:{url[:120]}"
    return hashlib.sha256(raw).hexdigest()
