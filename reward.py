"""
GRPO reward functions: correctness (primary) + thinking-length brevity (tie-break).

Target ordering (the spec):

    correct + short  >  correct + long  >  wrong + short  >  wrong + long

This is a *lexicographic* objective: correctness is the primary sort key and
thinking-brevity is the secondary key. We realize it as a weighted sum

    total = w_c * correctness + w_b * brevity   (+ w_f * format)

with the single guarantee condition  w_c > w_b .  Because correctness in {0,1}
and brevity in [0,1], `w_c > w_b` makes the *worst* correct sample (1 * w_c)
outrank the *best* wrong sample (1 * w_b), so brevity can never flip a
correct/incorrect ranking. With w_c=1.0, w_b=0.5:

    correct+short = 1.0 + 0.5*~1 = ~1.5
    correct+long  = 1.0 + 0.5*0  =  1.0
    wrong+short   = 0.0 + 0.5*~1 = ~0.5
    wrong+long    = 0.0 + 0.5*0  =  0.0
"""

from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Format configuration
# ---------------------------------------------------------------------------
# These MUST match the actual thinking delimiters your model emits.
#   - Toy run: we instruct a small model to use <think>...</think>.
#   - Gemma 4: call set_think_delimiters("<|channel>thought", "<channel|>").
#     VERIFIED against the gemma-4-E2B-it tokenizer (chat_template.jinja line 240
#     emits '<|channel>thought\n'+reasoning+'\n<channel|>'; tokenizer_config.json
#     gives soc_token="<|channel>", eoc_token="<channel|>", and an x-regex of
#     <|channel>thought\n(?P<thinking>.*?)<channel|>). If these are wrong, the
#     brevity reward silently collapses to 0 for every sample.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_HASH_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def set_think_delimiters(open_tok: str, close_tok: str) -> None:
    """Switch the thinking delimiters at runtime (call once at startup).
    Toy:      set_think_delimiters("<think>", "</think>")
    Gemma 4:  set_think_delimiters("<|channel>thought", "<channel|>")
    """
    global THINK_OPEN, THINK_CLOSE, _THINK_RE
    THINK_OPEN, THINK_CLOSE = open_tok, close_tok
    _THINK_RE = re.compile(re.escape(open_tok) + r"(.*?)" + re.escape(close_tok), re.DOTALL)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def get_completion_text(completion) -> str:
    """Handle both standard (str) and conversational (list[dict]) completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def extract_thinking(text: str) -> Optional[str]:
    """Return the (stripped) thinking span, or None if the block is missing."""
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_final_answer(text: str) -> str:
    """Everything after the think block is the final answer (fallback: whole text)."""
    m = _THINK_RE.search(text)
    return (text[m.end():] if m else text).strip()


def _answer_number(text: str) -> Optional[str]:
    """Prefer an explicit '#### N' marker; otherwise take the last number."""
    m = _HASH_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def is_correct(pred_text: str, gold_text: str) -> bool:
    p, g = _answer_number(pred_text), _answer_number(gold_text)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p == g


# --- AI-safety-guard task: block / allow verdict --------------------------
_VERDICT_RE = re.compile(r"####\s*(block|allow)", re.IGNORECASE)


def extract_verdict(text: str) -> Optional[str]:
    """Pull the block/allow verdict — the LAST such word in the post-think tail
    (the final answer comes after the reasoning), then a '#### block|allow' marker,
    then the last keyword anywhere. Taking the *last* word matters in native mode,
    where the think delimiters are stripped so the whole completion is the tail and
    the reasoning itself may mention 'block'/'allow'."""
    tail = extract_final_answer(text)
    found = re.findall(r"\b(block|allow)\b", tail, re.IGNORECASE)
    if found:
        return found[-1].lower()
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).lower()
    found = re.findall(r"\b(block|allow)\b", text, re.IGNORECASE)
    return found[-1].lower() if found else None


def is_correct_verdict(pred_text: str, gold_label: str) -> bool:
    """Correctness for the guard task: predicted verdict == gold label."""
    v = extract_verdict(pred_text)
    return v is not None and v == str(gold_label).strip().lower()


def count_think_tokens(think: str, tokenizer=None) -> int:
    """Token count of the thinking span. Falls back to whitespace words if no
    tokenizer is given (keeps unit tests dependency-free)."""
    if not think:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(think, add_special_tokens=False))
    return len(think.split())


def _length_norm(n: int, max_tokens: int, free: int = 0) -> float:
    """Map a token count to [0,1]:  0 = short/good, 1 = long.
    `free` is a no-penalty budget so we don't punish the minimum necessary
    reasoning (set free=0 for a pure brevity signal)."""
    if n <= free:
        return 0.0
    if n >= max_tokens:
        return 1.0
    return (n - free) / (max_tokens - free)


# ---------------------------------------------------------------------------
# Reward factory
# ---------------------------------------------------------------------------
def build_reward_funcs(
    tokenizer=None,
    max_think_tokens: int = 512,
    free_think_tokens: int = 0,
    wrong_answer_length_mode: str = "shorter_better",  # or "longer_better"
    answer_key: str = "answer",
    is_correct_fn=None,   # task-specific correctness; e.g. is_correct_verdict for the guard
    length_source: str = "think",   # "think": measure the <think> block.
                                    # "completion": measure the whole completion — use when
                                    # the model's think delimiters are special tokens that
                                    # trl strips at decode (e.g. Gemma's <|channel>thought).
):
    """Returns (correctness_reward, brevity_reward, format_reward).

    Plug into TRL with reward_weights=[1.0, 0.5, 0.2] (w_c > w_b is the only
    hard requirement; format weight is orthogonal / bootstrapping only).

    wrong_answer_length_mode:
      - "shorter_better": wrong+short > wrong+long  (the requested spec; rewards
        failing fast — saves compute but reduces exploration on hard problems).
      - "longer_better":  wrong+long  > wrong+short (cosine-reward style; keeps
        exploration alive when the model is unsure). Correct answers always
        prefer shorter regardless of this flag.
    """

    _correct = is_correct_fn or is_correct

    def correctness_reward(prompts=None, completions=None, **kwargs) -> List[float]:
        gold = kwargs[answer_key]
        out = []
        for comp, g in zip(completions, gold):
            text = extract_final_answer(get_completion_text(comp))
            out.append(1.0 if _correct(text, g) else 0.0)
        return out

    def brevity_reward(prompts=None, completions=None, **kwargs) -> List[float]:
        gold = kwargs[answer_key]
        out = []
        for comp, g in zip(completions, gold):
            text = get_completion_text(comp)
            if length_source == "completion":
                span = text          # whole completion ≈ reasoning (+ a one-word verdict)
            else:
                span = extract_thinking(text)
                if not span:         # missing/empty think block -> no brevity bonus
                    out.append(0.0)
                    continue
            norm = _length_norm(count_think_tokens(span, tokenizer), max_think_tokens, free_think_tokens)
            correct = _correct(extract_final_answer(text), g)
            out.append((1.0 - norm) if (correct or wrong_answer_length_mode == "shorter_better") else norm)
        return out

    def format_reward(prompts=None, completions=None, **kwargs) -> List[float]:
        # Small bonus for a well-formed think block. Constant across the 4 target
        # cases, so it does not affect their ordering; it only pushes the model
        # to adopt the format early (so brevity becomes measurable at all).
        return [1.0 if extract_thinking(get_completion_text(c)) else 0.0 for c in completions]

    return correctness_reward, brevity_reward, format_reward
