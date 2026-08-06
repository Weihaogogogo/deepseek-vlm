"""Merges VLM outputs and the user question into one image-info text block."""

LABEL_OVERALL = "【图片·整体】"
LABEL_FOCUS = "【图片·重点】"
LABEL_JUDGMENT = "【图片·判断】"
LABEL_QUESTION = "【用户问题】"

_MAX_IMAGES = 4


def merge_image_info(
    overall: str,
    focus: str | None,
    question: str,
    judgment: str | None = None,
) -> str:
    """Single-image merge: overall -> focus (optional) -> judgment (optional)
    -> user question."""
    return merge_multi_image(
        [{"overall": overall, "focus": focus, "judgment": judgment}], question
    )


def merge_multi_image(images: list[dict], question: str) -> str:
    """Multi-image merge. Each image dict:
    {"overall": str, "focus": str|None, "judgment": str|None}.

    Layout:
        【图片·整体1】 ... 【图片·重点1】 【图片·判断1】 【图片·整体2】 ... 【用户问题】
    Single-image output stays identical to merge_image_info (no numbering),
    so cached single-image descriptions keep their shape.
    judgment missing/empty is omitted (old cache entries have no judgment).
    """
    parts = []
    if len(images) == 1:
        img = images[0]
        parts.append(f"{LABEL_OVERALL}\n{img['overall'].strip()}")
        if img.get("focus"):
            parts.append(f"{LABEL_FOCUS}\n{img['focus'].strip()}")
        if img.get("judgment"):
            parts.append(f"{LABEL_JUDGMENT}\n{img['judgment'].strip()}")
    else:
        for idx, img in enumerate(images, start=1):
            parts.append(f"{LABEL_OVERALL}{idx}\n{img['overall'].strip()}")
            if img.get("focus"):
                parts.append(f"{LABEL_FOCUS}{idx}\n{img['focus'].strip()}")
            if img.get("judgment"):
                parts.append(f"{LABEL_JUDGMENT}{idx}\n{img['judgment'].strip()}")
    parts.append(f"{LABEL_QUESTION}\n{question.strip()}")
    return "\n\n".join(parts)
