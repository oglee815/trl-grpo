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


def test_guard_verdict():
    # AI-safety-guard task: correctness = predicted block/allow matches gold.
    from reward import is_correct_verdict
    reward.set_think_delimiters("<think>", "</think>")
    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=None, max_think_tokens=50,
        answer_key="label", is_correct_fn=is_correct_verdict,
    )
    mk = lambda n, v: f"<think>{' '.join(['reason'] * n)}</think>\n{v}"  # bare verdict (real format)
    cases = {
        "correct+short": mk(3, "block"), "correct+long": mk(45, "block"),
        "wrong+short":   mk(3, "allow"), "wrong+long":   mk(45, "allow"),
    }
    comps, gold = list(cases.values()), ["block"] * 4
    cr = correctness(completions=comps, label=gold)
    br = brevity(completions=comps, label=gold)
    fr = fmt(completions=comps, label=gold)
    total = [W_C * c + W_B * b + W_F * f for c, b, f in zip(cr, br, fr)]
    print("\n[guard block/allow]")
    print(f"{'case':14s} {'correct':>8s} {'brevity':>8s} {'TOTAL':>7s}")
    for name, c, b, t in zip(cases, cr, br, total):
        print(f"{name:14s} {c:8.2f} {b:8.3f} {t:7.3f}")
    assert total[0] > total[1] > total[2] > total[3], "GUARD ORDERING VIOLATED"
    print("OK  guard: correct+short > correct+long > wrong+short > wrong+long")
    # real Gemma output: bare verdict right after the thought channel close
    reward.set_think_delimiters("<|channel>thought", "<channel|>")
    assert reward.extract_verdict("<|channel>thought\nreasoning here<channel|>block") == "block"
    assert reward.extract_verdict("<|channel>thought\nlooks fine<channel|>allow") == "allow"
    reward.set_think_delimiters("<think>", "</think>")
    print("OK  guard: real channel format parses (<channel|>block -> block)")


def test_target_length_mode():
    """target_length>0: a correct answer's length reward peaks AT the target and
    falls off on both sides, so:  correct@target > correct@short > correct@long
    (3 is nearer to 20 than 45 is) > wrong.  Correctness still dominates."""
    from reward import is_correct_verdict
    reward.set_think_delimiters("<think>", "</think>")
    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=None, max_think_tokens=50,
        target_length=20, length_tolerance=50,
        answer_key="label", is_correct_fn=is_correct_verdict,
    )
    mk = lambda n, v: f"<think>{' '.join(['reason'] * n)}</think>\n{v}"
    cases = {
        "correct@target": mk(20, "block"),  # n == target  -> 1.0
        "correct@short":  mk(3,  "block"),  # 17 below target
        "correct@long":   mk(45, "block"),  # 25 above target
        "wrong@target":   mk(20, "allow"),  # right length, wrong verdict
    }
    comps, gold = list(cases.values()), ["block"] * 4
    cr = correctness(completions=comps, label=gold)
    br = brevity(completions=comps, label=gold)
    fr = fmt(completions=comps, label=gold)
    total = [W_C * c + W_B * b + W_F * f for c, b, f in zip(cr, br, fr)]
    print("\n[target-length mode]")
    print(f"{'case':16s} {'correct':>8s} {'brevity':>8s} {'TOTAL':>7s}")
    for name, c, b, t in zip(cases, cr, br, total):
        print(f"{name:16s} {c:8.2f} {b:8.3f} {t:7.3f}")
    assert total[0] > total[1] > total[2] > total[3], "TARGET-LENGTH ORDERING VIOLATED"
    print("OK  target: correct@target > correct@short > correct@long > wrong")
    reward.set_think_delimiters("<think>", "</think>")


def test_policy_scaled_length():
    """target_ratio>0: the target reasoning length scales with policy length, so
    the SAME reasoning is rewarded MORE under a long policy (which warrants long
    reasoning) than under a short policy (which should be read concisely)."""
    from reward import is_correct_verdict
    reward.set_think_delimiters("<think>", "</think>")
    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=None, max_think_tokens=200,
        target_ratio=0.5, tolerance_ratio=1.0,   # target=0.5*policy_len, tol=1.0*policy_len
        answer_key="label", is_correct_fn=is_correct_verdict,
    )
    comp = f"<think>{' '.join(['reason'] * 40)}</think>\nblock"   # 40-token reasoning (correct)
    br_long = brevity(completions=[comp], label=["block"], policy_len=[80])   # target=40 -> hit
    br_short = brevity(completions=[comp], label=["block"], policy_len=[20])  # target=10 -> far
    print("\n[policy-scaled length]")
    print(f"  same 40-tok reasoning:  long-policy(len=80,target=40) brevity={br_long[0]:.3f}")
    print(f"                          short-policy(len=20,target=10) brevity={br_short[0]:.3f}")
    assert br_long[0] > br_short[0], "POLICY-SCALED TARGET VIOLATED"
    print("OK  policy-scaled: longer policy rewards longer reasoning more than a short policy does")
    reward.set_think_delimiters("<think>", "</think>")


if __name__ == "__main__":
    test_toy_format()
    test_gemma4_channel_format()
    test_guard_verdict()
    test_target_length_mode()
    test_policy_scaled_length()
    print("\nALL OK")
