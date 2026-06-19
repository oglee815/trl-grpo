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

## AI-safety-guard task (the real goal)
Given a POLICY (with Block/Allow conditions) + INPUT, the model reasons then
outputs a single word `block`/`allow`. The prompt mirrors the real Gemma format:
the Role/Task/Decision-process instruction + `<policy_set><policy id="1">` +
`<input>` + `# Final Answer:`; the verdict is a **bare word right after the think
block** (no `####`). Correctness = verdict matches gold (`reward.extract_verdict`
/ `is_correct_verdict`, via `build_reward_funcs(is_correct_fn=...)`). Toy data
`guard_data.py`: 30 examples / 5 policies, each with a Block Condition and an
Allow Condition (exception), including "block-topic but allow-exception" cases so
the model must reason rather than keyword-match.

`train_guard.py` is a TRL `TrlParser` CLI (GuardArgs + GRPOConfig + ModelConfig),
launched via `train.sh` (multi-GPU `torchrun`) or directly. Local toy (Qwen, LoRA,
`<think>`):

```powershell
.\.venv\Scripts\python.exe train_guard.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct --use_peft --lora_target_modules all-linear `
  --num_generations 8 --per_device_train_batch_size 8 --learning_rate 1e-5 --temperature 0.7 `
  --max_completion_length 512 --max_think_tokens 256 --max_steps 30 --bf16 True `
  --report_to none --output_dir out-grpo-guard `
  --think_open "<think>" --think_close "</think>" --attn_implementation eager
```
Observed: policy-grounded reasoning in the think block then a bare `block`/`allow`
(`format≈1.0`, `brevity≈0.88`); correctness runs ~0.75–1.0 — a *real* signal,
since the allow-exception cases are genuinely harder. Reward tracks the 4-way
design (correct+concise ≈ 1.65, drops to ~1.4 when wrong).

Real data: a JSONL file — fields `input`, `label` (`block`/`allow`), and `policies`
(list[str]) or `policy` (str); see [guard_sample.jsonl](guard_sample.jsonl). Pass
`--dataset_name path.jsonl` (multiple `<policy>` per example is rendered into the
`<policy_set>`). GRPO needs **no gold reasoning** — only the verdict label.

B300 (8 GPUs, Gemma 4) — `train.sh` mirrors the `sft.py` workflow:

```bash
bash train.sh --model /path/to/gemma-4-E2B-it --data dataset/guard_train.jsonl \
  --epochs 3 --gpus 8 --batch 1 --accum 8 --num_generations 16 --native
```
`--native` turns on the Gemma thought channel (prompt pre-rendered with
`enable_thinking=True` → system `<|think|>`, verbatim single-word output); channel
delimiters `<|channel>thought` / `<channel|>` are the defaults. `num_generations`
must divide `gpus*batch*accum`. (If completions come out empty/garbled with
gradient checkpointing, add `--no_grad_ckpt`.) In `--native` mode brevity is
measured over the **whole completion length** — Gemma's thought-channel delimiters
are special tokens that trl strips at decode, so a `<think>` block can't be parsed
in the reward — and `max_think_tokens` is auto-raised to `max_completion_length`
(otherwise long completions all saturate brevity to 0 → no reward variance →
`loss=0`). The verdict is still parsed (it's the last `block`/`allow` word).

Two gotchas:
- Gold column must be `answer`, **not** `label` — trl reserves `label` (KeyError otherwise).
- On a non-thinking toy model (Qwen), the verbatim real instruction "Output only a
  single word" makes it skip reasoning (`format=0`, nothing for GRPO to reinforce).
  The toy Output Format elicits `<think>` explicitly; on Gemma 4 use the verbatim
  `GEMMA_OUTPUT_FORMAT` + native `<|think|>` / `enable_thinking=True`.

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
