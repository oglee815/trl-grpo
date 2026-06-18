"""Dependency-free checks that the reward ordering holds for BOTH the toy format
and the real Gemma 4 thought-channel format:
    correct+short > correct+long > wrong+short > wrong+long
Run:  python test_reward.py
"""
import reward
from reward import build_reward_funcs

W_C, W_B, W_F = 1.0, 0.5, 0.2  # correctness, brevity, format weights


def _score(comps, gold):
    correctness, brevity, fmt = build_reward_funcs(tokenizer=None, max_think_tokens=50)
    cr = correctness(completions=comps, answer=gold)
    br = brevity(completions=comps, answer=gold)
    fr = fmt(completions=comps, answer=gold)
    return cr, br, fr, [W_C * c + W_B * b + W_F * f for c, b, f in zip(cr, br, fr)]


def _report(label, cases, gold):
    comps = list(cases.values())
    cr, br, fr, total = _score(comps, gold)
    print(f"\n[{label}]")
    print(f"{'case':14s} {'correct':>8s} {'brevity':>8s} {'format':>7s} {'TOTAL':>7s}")
    for name, c, b, f, t in zip(cases, cr, br, fr, total):
        print(f"{name:14s} {c:8.2f} {b:8.3f} {f:7.2f} {t:7.3f}")
    assert total[0] > total[1] > total[2] > total[3], f"ORDERING VIOLATED ({label})"
    print(f"OK  {label}: correct+short > correct+long > wrong+short > wrong+long")


def test_toy_format():
    reward.set_think_delimiters("<think>", "</think>")
    mk = lambda n, a: f"<think>{' '.join(['reason'] * n)}</think>\nans\n#### {a}"
    _report("toy <think>", {
        "correct+short": mk(3, 42), "correct+long": mk(45, 42),
        "wrong+short": mk(3, 7),    "wrong+long":  mk(45, 7),
    }, ["#### 42"] * 4)


def test_gemma4_channel_format():
    # The real B300 target format (verify against the tokenizer's chat template).
    reward.set_think_delimiters("<|channel>thought", "<channel|>")
    mk = lambda n, a: f"<|channel>thought\n{' '.join(['reason'] * n)}<channel|>\nFinal answer.\n#### {a}"
    _report("gemma4 channel", {
        "correct+short": mk(3, 42), "correct+long": mk(45, 42),
        "wrong+short": mk(3, 7),    "wrong+long":  mk(45, 7),
    }, ["#### 42"] * 4)
    # show extraction is actually pulling the right spans
    sample = mk(3, 42)
    print("  extracted think :", repr(reward.extract_thinking(sample)))
    print("  extracted answer:", repr(reward.extract_final_answer(sample)))
    reward.set_think_delimiters("<think>", "</think>")  # reset


if __name__ == "__main__":
    test_toy_format()
    test_gemma4_channel_format()
    print("\nALL OK")
