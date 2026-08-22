from feed.providers.reasoning import strip_reasoning


def test_strips_a_realistic_think_block():
    """Verified live: qwen/qwen3.6-27b emits visible <think>...</think> in
    its output even when the prompt does not ask for it."""
    raw = (
        "<think>\n"
        "The user wants a JSON object with headline, summary, category. "
        "Let me draft it: this is about a new model release from an AI lab. "
        "I'll keep the summary to two sentences as instructed.\n"
        "</think>\n"
        '{"headline": "Lab ships new model", '
        '"summary": "A lab released a new model today. It improves on the '
        'prior version.", "category": "product"}'
    )

    result = strip_reasoning(raw)

    assert "<think>" not in result
    assert "</think>" not in result
    assert "draft it" not in result
    assert result == (
        '{"headline": "Lab ships new model", '
        '"summary": "A lab released a new model today. It improves on the '
        'prior version.", "category": "product"}'
    )


def test_strips_multiple_reasoning_tag_names():
    for tag in ("think", "thinking", "reasoning", "reflection"):
        raw = f"<{tag}>internal chatter</{tag}>final answer"
        assert strip_reasoning(raw) == "final answer"


def test_leaves_plain_text_untouched():
    assert strip_reasoning("just a normal answer") == "just a normal answer"


def test_strips_leading_and_trailing_whitespace_left_by_removal():
    raw = "  <think>noise</think>  \n\nactual content  "
    assert strip_reasoning(raw) == "actual content"


def test_handles_an_unclosed_reasoning_block_from_a_truncated_response():
    """A response cut off by a token limit mid-thought must not leak the
    dangling reasoning fragment into what gets stored."""
    raw = "<think>\nStill reasoning about this and it just cuts off"
    assert strip_reasoning(raw) == ""


def test_content_before_an_unclosed_block_is_kept():
    raw = "final answer here<think>and then it trails off with no close"
    assert strip_reasoning(raw) == "final answer here"


def test_empty_and_none_like_input_is_handled():
    assert strip_reasoning("") == ""


def test_is_case_insensitive():
    raw = "<THINK>noise</THINK>answer"
    assert strip_reasoning(raw) == "answer"
