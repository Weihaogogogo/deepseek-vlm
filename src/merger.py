"""Merges VLM outputs and the user question into one image-info text block."""

LABEL_OVERALL = "【图片·整体】"
LABEL_FOCUS = "【图片·重点】"
LABEL_QUESTION = "【用户问题】"

_MAX_IMAGES = 4


def merge_image_info(overall: str, focus: str | None, question: str) -> str:
    """Single-image merge: overall -> focus (optional) -> user question."""
    return merge_multi_image(
        [{"overall": overall, "focus": focus}], question
    )


def merge_multi_image(images: list[dict], question: str) -> str:
    """Multi-image merge. Each image dict: {"overall": str, "focus": str|None}.

    Layout:
        【图片·整体1】 ... 【图片·重点1】 【图片·整体2】 ... 【用户问题】
    Single-image output stays identical to merge_image_info (no numbering),
    so cached single-image descriptions keep their shape.
    """
    parts = []
    if len(images) == 1:
        img = images[0]
        parts.append(f"{LABEL_OVERALL}\n{img['overall'].strip()}")
        if img.get("focus"):
            parts.append(f"{LABEL_FOCUS}\n{img['focus'].strip()}")
    else:
        for idx, img in enumerate(images, start=1):
            parts.append(f"{LABEL_OVERALL}{idx}\n{img['overall'].strip()}")
            if img.get("focus"):
                parts.append(f"{LABEL_FOCUS}{idx}\n{img['focus'].strip()}")
    parts.append(f"{LABEL_QUESTION}\n{question.strip()}")
    return "\n\n".join(parts)
