# trl-grpo: length-aware reasoning reward (GRPO)

Reward objective (lexicographic — correctness first, brevity as tie-break):

```
correct + short  >  correct + long  >  wrong + short  >  wrong + long
```

Realized as `total = w_c*correct + w_b*brevity + w_f*format` with the single hard
rule **`w_c > w_b`** (so brevity can never flip a correct/incorrect ranking). See
[reward.py](reward.py); the 4-way ordering is asserted in [test_reward.py](test_reward.py)
for both the toy `<think>` format and the real Gemma 4 thought-channel format.

## Files
- `reward.py` — parsing + 3 reward fns (correctness / brevity / format). `set_think_delimiters()` switches formats.
- `test_reward.py` — proves the ordering (no GPU/torch needed): `python test_reward.py`
- `train_grpo.py` — TRL GRPO toy loop.

## Local Windows toy (verified)
Smallest mobile Gemma available now is `gemma-3-270m-it` (Gemma 4 has no <1B size;
its smallest is E2B ~5B, which needs QLoRA on a 12GB GPU). Two Windows-only fixes:

1. **`PYTHONUTF8=1`** — trl reads bundled `*.jinja` chat templates with the OS
   locale codec (cp949 on Korean Windows) and crashes; UTF-8 mode fixes it.
2. **trl/torch version skew** — global `trl 1.5.1` imports `FSDPModule` (needs
   torch ≥ 2.4) but this box has `torch 2.3.0+cu121`. We isolate a compatible trl
   in a venv that inherits the global packages, leaving the global env untouched:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install "trl==0.19.1" "peft" --no-deps   # newest trl that imports on torch 2.3
$env:PYTHONUTF8="1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
.\.venv\Scripts\python.exe train_grpo.py
```
Restore global trl anytime by deleting `.venv`.

3. **Keep gradient checkpointing OFF** (`GRAD_CKPT=0`, the default). With
   `trl 0.19`, GC leaks into the *generation* phase (forces `use_cache=False`)
   and corrupts every completion — output is garbage, length pins at the cap
   (`clipped_ratio=1.0`), and all rewards read 0. This wasted two training runs
   before we caught it by dumping a completion. Re-enable GC only on the matched
   B300 stack (trl 1.5 + torch ≥ 2.5).

### Seeing the brevity reward actually move
`gemma-3-270m-it` is too weak to emit the format (completions never close the
think block → brevity/format stay 0; the reward *logic* is still proven by
`test_reward.py`). A small instruct model that follows `<think>` shows the signal
live — LoRA fits the 12GB GPU:

```powershell
$env:MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"; $env:USE_LORA="1"
$env:NUM_GEN="8"; $env:MAX_STEPS="20"; $env:LR="1e-5"; $env:TEMP="0.7"; $env:MAX_COMPLETION="512"
.\.venv\Scripts\python.exe train_grpo.py
```
Observed: `format≈1.0`, `brevity≈0.75–0.97` with per-step variance, `clip=0.0`
(all terminate via EOS), `grad_norm>0` every step. When a correct answer appears
the reward jumps ~+1.0 over the wrong-but-short baseline (~0.67) — the 4-way
ordering operating live. `check_format.py` pre-flights whether a model emits a
closable think block before you spend a training run; `parse_log.py` tabulates a
saved stdout log.

## B300 (Linux) real run — checklist
- `MODEL_NAME = "google/gemma-4-E2B-it"` (smallest Gemma 4; E4B for more quality).
- In `train_grpo.py`, call once: `reward.set_think_delimiters("<|channel>thought", "<channel|>")`
  and **verify** against `tokenizer.apply_chat_template(...)` output — the exact
  channel tokens are the #1 risk; if wrong, brevity silently reads 0.
- Enable thinking via the `<|think|>` control token in the system prompt.
- `use_vllm=True`, `attn_implementation="flash_attention_2"`.
- Raise `max_completion_length` (e.g. 1024–4096) so the thought channel can close
  and terminate with EOS (toy showed all completions truncated).
- `num_generations` 8–16; global trl 1.5.x is fine there (torch ≥ 2.4).
- Tuning: `wrong_answer_length_mode` ("shorter_better" = spec; "longer_better" =
  preserve exploration on hard problems), `free_think_tokens` (no-penalty budget),
  and consider hardening correctness to require the `#### N` marker (the last-number
  fallback can score garbled output as correct).
```
