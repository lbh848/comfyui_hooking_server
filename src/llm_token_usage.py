"""Token estimates shared by live streams and stored LLM logs."""


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 3) if text else 0


def output_token_stats(
    text: str,
    reasoning_text: str = "",
    *,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    reasoning_chars: int = 0,
) -> dict:
    """Completion usage includes reasoning; never add it to an existing total."""
    fields = {}
    chars = len(reasoning_text) or reasoning_chars
    if reasoning_tokens is None and chars > 0:
        reasoning_tokens = max(1, chars // 3)
        if completion_tokens is not None:
            reasoning_tokens = min(reasoning_tokens, completion_tokens)
        fields["reasoning_tokens_estimated"] = True
    elif reasoning_tokens is not None:
        fields["reasoning_tokens_estimated"] = False
    if reasoning_tokens is not None:
        fields["reasoning_tokens"] = reasoning_tokens
    fields["tokens_estimated"] = completion_tokens is None
    fields["completion_tokens"] = (
        approx_tokens(text) + (reasoning_tokens or 0)
        if completion_tokens is None else completion_tokens
    )
    return fields
