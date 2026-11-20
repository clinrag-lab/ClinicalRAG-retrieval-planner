"""Token counting. All token numbers in the project go through count_tokens."""

from __future__ import annotations

# Rough tokens-per-word ratio for English clinical text, as a fraction (13/10 = 1.3).
# Integer arithmetic avoids float rounding (10 * 1.3 = 13.000000000000002 -> ceil 14).
_TOKENS_PER_WORD_NUM = 13
_TOKENS_PER_WORD_DEN = 10


def count_tokens(text: str) -> int:
    """Approximate LLM tokens in text: whitespace word count * 1.3, rounded up.

    Replace the body of this function to switch to a real tokenizer.
    """
    words = len(text.split())
    return -(-words * _TOKENS_PER_WORD_NUM // _TOKENS_PER_WORD_DEN)
