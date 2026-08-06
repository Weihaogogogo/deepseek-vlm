"""Merges VLM outputs into one image-info text block per image."""


def merge_image_block(overall: str, focus: str | None, judgment: str | None) -> str:
    """Single-image block text: overall -> focus (optional) -> judgment
    (optional), separated by '---'. Empty sections are omitted entirely."""
    parts = [overall.strip()] if overall.strip() else []
    if focus and focus.strip():
        parts.append(focus.strip())
    if judgment and judgment.strip():
        parts.append(judgment.strip())
    return "\n---\n".join(parts)


def merge_multi_image(images: list[dict]) -> list[dict]:
    """Multi-image merge. Each image dict:
    {"overall": str, "focus": str|None, "judgment": str|None}.

    Returns a list of {"type": "text", "text": ...} parts, one per image in
    sending order; the array index maps to the [图片 N] placeholder in the
    user message (block 0 <-> [图片 1]). judgment missing/empty is omitted
    (old cache entries have no judgment).
    """
    return [
        {
            "type": "text",
            "text": merge_image_block(
                img.get("overall", ""),
                img.get("focus"),
                img.get("judgment"),
            ),
        }
        for img in images
    ]
