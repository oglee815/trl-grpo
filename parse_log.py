"""Pretty-print per-step GRPO metrics from a saved stdout log.
Usage:  python parse_log.py [run_log.txt]
"""
import ast
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\workspace\claude\trl-grpo\run_log.txt"
raw = open(PATH, "rb").read()
enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
rows = []
for line in raw.decode(enc, errors="ignore").splitlines():
    line = line.strip()
    if line.startswith("{'loss'") and "completions/mean_length" in line:
        rows.append(ast.literal_eval(line))

hdr = f"{'step':>4} {'len_mean':>8} {'len_min':>7} {'len_max':>7} {'clip':>5} {'fmt':>4} {'brevity':>8} {'correct':>7} {'reward':>7} {'grad':>6}"
print(hdr)
print("-" * len(hdr))
for i, d in enumerate(rows, 1):
    print(f"{i:>4} {d['completions/mean_length']:>8.1f} {d['completions/min_length']:>7.0f} "
          f"{d['completions/max_length']:>7.0f} {d['completions/clipped_ratio']:>5.2f} "
          f"{d['rewards/format_reward/mean']:>4.2f} {d['rewards/brevity_reward/mean']:>8.3f} "
          f"{d['rewards/correctness_reward/mean']:>7.3f} {d['reward']:>7.3f} {d['grad_norm']:>6.2f}")


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


if len(rows) >= 6:
    k = max(3, len(rows) // 4)
    first, last = rows[:k], rows[-k:]
    lf, ll = avg([r['completions/mean_length'] for r in first]), avg([r['completions/mean_length'] for r in last])
    bf, bl = avg([r['rewards/brevity_reward/mean'] for r in first]), avg([r['rewards/brevity_reward/mean'] for r in last])
    print("-" * len(hdr))
    print(f"first {k} steps : len={lf:6.1f}  brevity={bf:.3f}")
    print(f"last  {k} steps : len={ll:6.1f}  brevity={bl:.3f}")
    print(f"delta          : len={ll - lf:+6.1f}  brevity={bl - bf:+.3f}")
