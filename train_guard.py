"""
GRPO training for the AI-safety-guard task — CLI / torchrun launcher.

Uses TRL's standard parser (GuardArgs + GRPOConfig + ModelConfig), so every HF /
GRPO / LoRA flag is available on the command line, and it runs under torchrun for
multi-GPU. Launch via train.sh, or directly:

  python train_guard.py --model_name_or_path <path> --dataset_name <jsonl> \
      --num_generations 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
      --learning_rate 1e-5 --num_train_epochs 3 --bf16 True --output_dir out --use_peft

B300 / Gemma 4: add  --native_thinking --think_open "<|channel>thought" \
      --think_close "<channel|>" --attn_implementation flash_attention_2
"""
import os
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

import reward as reward_mod
from reward import (build_reward_funcs, is_correct_verdict, get_completion_text,
                    extract_verdict, count_think_tokens)
from guard_data import get_guard_dataset


@dataclass
class GuardArgs:
    dataset_name: str = field(default="", metadata={"help": "JSONL path; empty = built-in toy set"})
    max_think_tokens: int = field(default=512, metadata={"help": "brevity normalizer"})
    brevity_weight: float = field(default=0.5, metadata={"help": "must stay < 1.0"})
    format_weight: float = 0.2
    target_length: int = field(default=0, metadata={
        "help": "if >0, a correct answer's length reward peaks at this many think "
                "tokens (closeness), instead of the default shorter-is-better"})
    length_tolerance: int = field(default=0, metadata={
        "help": "tokens away from target_length where that reward hits 0; "
                "0 = use max_think_tokens"})
    target_ratio: float = field(default=0.0, metadata={
        "help": "if >0, per-example target = target_ratio * policy tokens "
                "(scale the reasoning budget with policy length); overrides target_length"})
    tolerance_ratio: float = field(default=0.0, metadata={
        "help": "if >0, per-example tolerance = tolerance_ratio * policy tokens"})
    wrong_answer_length_mode: str = field(default="shorter_better",
                                          metadata={"help": "shorter_better | longer_better"})
    think_open: str = "<think>"
    think_close: str = "</think>"
    native_thinking: bool = field(default=False, metadata={
        "help": "Gemma native thought channel: pre-render the prompt with "
                "enable_thinking=True and use the verbatim single-word output format"})
    dump_completions: bool = field(default=False, metadata={"help": "dump one group of completions (debug)"})
    dump_every: int = field(default=0, metadata={"help": "dump every N reward batches; 0 = once at start"})
    dump_chars: int = field(default=0, metadata={"help": "chars of each completion to print; 0 = full text"})


GUARD_BODY = (
    "# Role\n"
    "You are a sophisticated policy-based classification model acting as an AI Safety Filter.\n\n"
    "# Task\n"
    "Analyze a given Input (a user query or an LLM/Agent response) against a Policy "
    "(one or more policy sets) and decide whether the Input should be Blocked or Allowed.\n\n"
    "# Decision and Reasoning Process\n"
    "1. If it does NOT match the Block Condition: Allow.\n"
    "2. If it matches the Block Condition BUT also satisfies the Allow Condition: Allow.\n"
    "3. If it matches the Block Condition AND does NOT satisfy the Allow Condition: Block.\n"
    "4. For multiple policies, apply the above to each; violating one or more is a total Block.\n\n"
    "# Notion\n"
    "- If block/allow conditions are not explicit, read the whole policy, interpret which "
    "statements are block vs allow conditions, then decide.\n"
    "- Keep your internal reasoning brief and concise."
)
# Gemma 4 (native thinking): the post-channel output is genuinely one word.
GEMMA_OUTPUT_FORMAT = "\n\n# Output Format\nOutput only a single word: block or allow."


def toy_output_format(think_open: str, think_close: str) -> str:
    # Non-thinking models (e.g. Qwen) skip reasoning under a bare "one word"
    # instruction, so we elicit the think wrapper explicitly.
    return (
        f"\n\n# Output Format (IMPORTANT)\nYour reply MUST begin with {think_open}, then brief "
        f"reasoning, then {think_close}, then a single word: block or allow. Do not write the "
        f"verdict before {think_open}.\n"
        f"Example:\n{think_open} The input asks for public facts; no block condition applies. "
        f"{think_close} allow"
    )


def _render_policy_set(policies):
    items = "\n".join(f'<policy id="{i}">\n{p}\n</policy>' for i, p in enumerate(policies, 1))
    return f"<policy_set>\n{items}\n</policy_set>"


def build_dataset(tokenizer, ga: GuardArgs):
    ds = get_guard_dataset(ga.dataset_name or None)
    out_fmt = GEMMA_OUTPUT_FORMAT if ga.native_thinking else toy_output_format(ga.think_open, ga.think_close)

    def fmt(ex):
        policy_str = _render_policy_set(ex["policies"])
        user = (
            GUARD_BODY + out_fmt
            + "\n\n-----\n\nNow decide for the policy and input below.\n\n"
            + f"# Policy:\n{policy_str}\n\n"
            + f"# Input:\n<input>\n{ex['input']}\n</input>\n\n# Final Answer:"
        )
        # policy token length -> used by the policy-scaled target/tolerance reward
        policy_len = len(tokenizer.encode("\n".join(ex["policies"]), add_special_tokens=False))
        messages = [{"role": "user", "content": user}]
        if ga.native_thinking:
            # Pre-render so the system turn carries <|think|>. trl tokenizes prompt
            # strings with add_special_tokens=False, so no double BOS.
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return {"prompt": prompt, "answer": ex["label"], "policy_len": policy_len}
        return {"prompt": messages, "answer": ex["label"], "policy_len": policy_len}  # toy

    return ds.map(fmt, remove_columns=ds.column_names)


def _apply_leftovers(leftover, training_args, model_args):
    """Apply CLI flags this trl version's parser didn't recognize, so the run does
    not crash on a renamed/removed field. Handles known renames (torch_dtype <->
    dtype) and warns + skips anything with no matching field."""
    aliases = {"torch_dtype": ("dtype", "torch_dtype"), "dtype": ("dtype", "torch_dtype")}
    pairs, i = [], 0
    while i < len(leftover):
        t = leftover[i]
        if not t.startswith("--"):
            i += 1
            continue
        if "=" in t:
            k, v, i = t[2:].split("=", 1)[0], t.split("=", 1)[1], i + 1
        elif i + 1 < len(leftover) and not leftover[i + 1].startswith("--"):
            k, v, i = t[2:], leftover[i + 1], i + 2
        else:
            k, v, i = t[2:], "true", i + 1
        pairs.append((k, v))

    def _cast(v, ref):
        if isinstance(ref, bool):
            return str(v).lower() in ("1", "true", "yes")
        if isinstance(ref, float):
            try:
                return float(v)
            except ValueError:
                return ref
        if isinstance(ref, int):
            try:
                return int(v)
            except ValueError:
                return ref
        return v

    for k, v in pairs:
        for obj in (training_args, model_args):
            hit = next((n for n in aliases.get(k, (k,)) if hasattr(obj, n)), None)
            if hit is not None:
                setattr(obj, hit, _cast(v, getattr(obj, hit)))
                print(f"[args] --{k} -> {type(obj).__name__}.{hit}={v}", flush=True)
                break
        else:
            print(f"[args] WARNING: --{k}={v} is not a field in this trl version; ignored", flush=True)


def main():
    guard_args, training_args, model_args, leftover = TrlParser(
        (GuardArgs, GRPOConfig, ModelConfig)
    ).parse_args_and_config(return_remaining_strings=True)
    _apply_leftovers(leftover, training_args, model_args)

    reward_mod.set_think_delimiters(guard_args.think_open, guard_args.think_close)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    eff_max_think = guard_args.max_think_tokens
    fmt_weight = guard_args.format_weight
    if guard_args.native_thinking:
        # Native-thinking models emit the thought channel with SPECIAL tokens, which
        # trl strips when decoding completions -> the <think> block can't be parsed.
        # Measure brevity over the whole completion instead, and drop the format
        # reward (a closable think block is neither observable nor needed here).
        fmt_weight = 0.0
        if eff_max_think < training_args.max_completion_length:
            eff_max_think = training_args.max_completion_length
            print(f"[guard] native: max_think_tokens -> {eff_max_think} "
                  f"(= max_completion_length) so brevity spans the full length range", flush=True)

    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=tokenizer,
        max_think_tokens=eff_max_think,
        target_length=guard_args.target_length,
        length_tolerance=guard_args.length_tolerance,
        target_ratio=guard_args.target_ratio,
        tolerance_ratio=guard_args.tolerance_ratio,
        wrong_answer_length_mode=guard_args.wrong_answer_length_mode,
        answer_key="answer",
        is_correct_fn=is_correct_verdict,
        length_source="completion" if guard_args.native_thinking else "think",
    )
    if guard_args.target_ratio > 0:
        tr = guard_args.tolerance_ratio or "length_tolerance"
        print(f"[guard] length reward = closeness to a POLICY-SCALED target "
              f"(target={guard_args.target_ratio}*policy_tok, tol={tr}*policy_tok) "
              f"for CORRECT answers", flush=True)
    elif guard_args.target_length > 0:
        tol = guard_args.length_tolerance or eff_max_think
        print(f"[guard] length reward = closeness to target_length={guard_args.target_length} "
              f"(±{tol} tok -> 0) for CORRECT answers", flush=True)
    reward_funcs = [correctness, brevity, fmt]
    training_args.reward_weights = [1.0, guard_args.brevity_weight, fmt_weight]

    if guard_args.dump_completions:
        _calls = [0]
        n_gen = training_args.num_generations
        every = guard_args.dump_every

        def debug_dump(prompts=None, completions=None, **kwargs):
            # Dump one whole group (num_generations completions of the SAME prompt) on
            # rank 0 — once at start (dump_every=0) or every N reward batches. If the
            # texts differ but verdict/length are uniform it's reward saturation (->
            # need harder data); if the texts are near-identical it's entropy collapse.
            if int(os.environ.get("RANK", "0")) != 0:
                return [0.0] * len(completions)
            i = _calls[0]
            _calls[0] += 1
            if not ((i == 0) if every <= 0 else (i % every == 0)):
                return [0.0] * len(completions)
            gold = kwargs.get("answer", [None] * len(completions))
            plen = kwargs.get("policy_len", [None] * len(completions))
            n = min(n_gen, len(completions))
            p = prompts[0]
            ptxt = p if isinstance(p, str) else (p[-1].get("content", "") if isinstance(p, list) and p else str(p))
            print("\n" + "=" * 72)
            print(f"[dump] batch {i}: one group of {n} completions | gold={gold[0]} | policy_len={plen[0]}")
            print(f"[dump] prompt tail: ...{ptxt[-160:]!r}")
            for j in range(n):
                t = get_completion_text(completions[j])
                ok = is_correct_verdict(t, gold[j]) if gold[j] is not None else "?"
                print(f"\n----- [{j:02d}] tok={count_think_tokens(t, tokenizer)} "
                      f"verdict={extract_verdict(t)} correct={ok} -----")
                print(t if guard_args.dump_chars <= 0 else t[:guard_args.dump_chars])
            print("=" * 72, flush=True)
            return [0.0] * len(completions)

        reward_funcs.append(debug_dump)
        training_args.reward_weights.append(0.0)

    import transformers
    mik = {}
    if model_args.attn_implementation:
        mik["attn_implementation"] = model_args.attn_implementation
    dt = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    if dt and dt != "auto":  # transformers 5 renamed model_init torch_dtype -> dtype
        mik["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = dt
    training_args.model_init_kwargs = mik
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    peft_config = get_peft_config(model_args)
    if peft_config is not None and peft_config.target_modules == ["all-linear"]:
        peft_config.target_modules = "all-linear"   # peft wants the bare string

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=build_dataset(tokenizer, guard_args),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
