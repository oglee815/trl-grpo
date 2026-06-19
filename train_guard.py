"""
GRPO toy for the AI-safety-guard task: given a POLICY and an INPUT, decide
'block' (input violates policy) or 'allow', thinking first.

Reward ordering (same lexicographic design as the math toy):
    correct + short  >  correct + long  >  wrong + short  >  wrong + long
where "correct" = predicted block/allow verdict matches the gold label.

Local demo (Windows, 12GB):
  $env:MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"; $env:USE_LORA="1"
  $env:NUM_GEN="8"; $env:MAX_STEPS="20"; $env:LR="1e-5"; $env:TEMP="0.7"; $env:MAX_COMPLETION="512"
  .\.venv\Scripts\python.exe train_guard.py
B300: MODEL_NAME=google/gemma-4-E2B-it + set_think_delimiters("<|channel>thought","<channel|>")
      + enable_thinking=True prerender (see train_grpo.py header).
"""
import os

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import build_reward_funcs, is_correct_verdict, THINK_OPEN, THINK_CLOSE
from guard_data import build_guard_dataset

# --- Config (toy defaults) -------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-270m-it")
MAX_THINK_TOKENS = int(os.environ.get("MAX_THINK", "256"))
USE_VLLM = False
USE_LORA = os.environ.get("USE_LORA", "0") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
NUM_GEN = int(os.environ.get("NUM_GEN", "4"))
LR = float(os.environ.get("LR", "1e-6"))
TEMP = float(os.environ.get("TEMP", "1.0"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "256"))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"   # keep OFF (see train_grpo.py note)
BREVITY_W = float(os.environ.get("BREVITY_W", "0.5"))  # must stay < 1.0

# Real guard instruction body (Role / Task / Decision / Notion). The Output Format
# section differs by model and is appended in build_dataset().
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

# Real Gemma 4: native thinking (system <|think|> + enable_thinking=True), so the
# post-channel output is genuinely a single word. Use this verbatim on B300.
GEMMA_OUTPUT_FORMAT = "\n\n# Output Format\nOutput only a single word: block or allow."

# Toy: Qwen is NOT a native thinking model, so a bare "output one word" makes it
# skip reasoning entirely (format=0 for every sample -> GRPO has nothing to
# reinforce). The toy Output Format elicits the <think> wrapper explicitly.
# Swap for GEMMA_OUTPUT_FORMAT on B300.
TOY_OUTPUT_FORMAT = (
    f"\n\n# Output Format (IMPORTANT)\nYour reply MUST begin with {THINK_OPEN}, then brief "
    f"reasoning, then {THINK_CLOSE}, then a single word: block or allow. Do not write the "
    f"verdict before {THINK_OPEN}.\n"
    f"Example:\n{THINK_OPEN} The input asks for public facts; no block condition applies. "
    f"{THINK_CLOSE} allow"
)


def build_dataset():
    ds = build_guard_dataset()

    def fmt(ex):
        user = (
            GUARD_BODY + TOY_OUTPUT_FORMAT
            + "\n\n-----\n\nNow decide for the policy and input below.\n\n"
            + f'# Policy:\n<policy_set>\n<policy id="1">\n{ex["policy"]}\n</policy>\n</policy_set>\n\n'
            + f"# Input:\n<input>\n{ex['input']}\n</input>\n\n# Final Answer:"
        )
        # NOTE: name the gold column "answer", NOT "label" — trl reserves "label"
        # and would try to validate it as a chat-template key (KeyError otherwise).
        return {"prompt": [{"role": "user", "content": user}], "answer": ex["label"]}

    return ds.map(fmt, remove_columns=ds.column_names)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=tokenizer,
        max_think_tokens=MAX_THINK_TOKENS,
        wrong_answer_length_mode="shorter_better",
        answer_key="answer",                # gold column ("label" is reserved by trl)
        is_correct_fn=is_correct_verdict,   # guard correctness = verdict match
    )

    reward_funcs = [correctness, brevity, fmt]
    reward_weights = [1.0, BREVITY_W, 0.2]
    if os.environ.get("DEBUG") == "1":
        from reward import get_completion_text, extract_verdict
        _seen = []

        def debug_dump(prompts=None, completions=None, **kwargs):
            if not _seen:
                txt = get_completion_text(completions[0])
                print("\n===== DEBUG completion[0] =====")
                print(repr(txt[:1200]))
                print("verdict ->", extract_verdict(txt), "\n", flush=True)
                _seen.append(1)
            return [0.0] * len(completions)

        reward_funcs.append(debug_dump)
        reward_weights.append(0.0)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_init_kwargs = {"attn_implementation": "eager"}
    if USE_LORA:
        model_init_kwargs["torch_dtype"] = "bfloat16"

    peft_config = None
    if USE_LORA:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    args = GRPOConfig(
        output_dir="out-grpo-guard",
        per_device_train_batch_size=NUM_GEN,
        gradient_accumulation_steps=1,
        num_generations=NUM_GEN,
        max_prompt_length=384,
        max_completion_length=MAX_COMPLETION,
        temperature=TEMP,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="no",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=USE_LORA and GRAD_CKPT,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=USE_VLLM,
        reward_weights=reward_weights,
        model_init_kwargs=model_init_kwargs,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=build_dataset(),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    print("guard toy run finished -> ./out-grpo-guard")


if __name__ == "__main__":
    main()
