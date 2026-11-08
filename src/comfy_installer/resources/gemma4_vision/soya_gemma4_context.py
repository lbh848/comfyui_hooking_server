"""Fit explicitly marked past conversation to the runtime's exact token budget."""

import re


def fit_history(prompt, history_ranges, count_budget):
    """Keep recent complete turns; never edit text outside the supplied ranges."""
    spans, groups = [], {}
    previous_end = 0
    for span in history_ranges:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(value) is not int for value in span)):
            raise ValueError("Invalid Gemma history range.")
        start, end = span
        if not previous_end <= start < end <= len(prompt):
            raise ValueError("Invalid Gemma history range bounds.")
        previous_end = end
        text = prompt[start:end]
        if not text.strip() or text.strip() == "(none)":
            continue
        # Combined scene/Key Visual calls can contain the same history twice.
        # Give both copies the same retained suffix.
        boundaries = [0, *(match.start() for match in re.finditer(
            r"(?m)^\[(?:CHAR|USER)\]\r?$", text) if match.start() > 0), len(text)]
        group = (text, None if len(boundaries) > 2 else start)
        groups.setdefault(group, list(zip(boundaries, boundaries[1:])))
        spans.append((start, end, group))

    turns = [(text, end) for text, chunks in groups.items() for _, end in chunks]

    def render(removed):
        cuts = {}
        for text, end in turns[:removed]:
            cuts[text] = end
        parts, cursor, kept_chars = [], 0, 0
        for start, end, group in spans:
            retained = group[0][cuts.get(group, 0):]
            parts.extend((prompt[cursor:start], retained or "(none)"))
            cursor = end
            kept_chars += len(retained)
        parts.append(prompt[cursor:])
        return "".join(parts), kept_chars

    def measure(text):
        budget = count_budget(text)
        if budget["input_tokens"] is None or budget["context_limit"] is None:
            raise RuntimeError("Gemma 입력 토큰 수 또는 컨텍스트 한도를 확인하지 못했습니다.")
        return budget

    original = measure(prompt)
    budget, adjusted, removed = original, prompt, 0
    before = kept_chars = sum(end - start for start, end, _ in spans)
    if budget["remaining"] < 0:
        adjusted, kept_chars = render(len(turns))
        budget = measure(adjusted) if turns else original
        if budget["remaining"] < 0:
            raise RuntimeError(
                f"Gemma 컨텍스트 한도 초과: 히스토리를 제외한 필수 입력 {budget['input_tokens']:,}토큰과 "
                f"응답 한도 {budget['max_tokens']:,}토큰이 컨텍스트 {budget['context_limit']:,}토큰을 초과합니다. "
                "'컨텍스트'를 늘리거나 현재 입력을 줄이세요."
            )
        # Search only whole-turn suffixes, measuring the complete final payload.
        low, high = 1, len(turns)
        removed = high
        while low < high:
            middle = (low + high) // 2
            candidate, chars = render(middle)
            candidate_budget = measure(candidate)
            if candidate_budget["remaining"] >= 0:
                high = middle
                adjusted, kept_chars, budget, removed = candidate, chars, candidate_budget, middle
            else:
                low = middle + 1
    return adjusted, {
        "input_tokens_before": original["input_tokens"], "input_tokens": budget["input_tokens"],
        "context_limit": budget["context_limit"], "max_tokens": budget["max_tokens"],
        "history_chars_before": before, "history_chars": kept_chars,
        "history_turns_removed": removed,
    }
