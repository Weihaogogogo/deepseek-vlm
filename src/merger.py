"""Merges VLM outputs and the user question into one image-info text block."""

LABEL_OVERALL = "【图片·整体】"
LABEL_FOCUS = "【图片·重点】"
LABEL_QUESTION = "【用户问题】"


def merge_image_info(overall: str, focus: str | None, question: str) -> str:
    """Pure concatenation: overall -> focus (optional) -> user question."""
    parts = [f"{LABEL_OVERALL}\n{overall.strip()}"]
    if focus:
        parts.append(f"{LABEL_FOCUS}\n{focus.strip()}")
    parts.append(f"{LABEL_QUESTION}\n{question.strip()}")
    return "\n\n".join(parts)
