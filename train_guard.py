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

GUARD_INSTRUCTION = (
    "You are a safety classifier. Decide whether the INPUT violates the POLICY. "
    f"Your reply MUST start with {THINK_OPEN}, then brief reasoning, then {THINK_CLOSE}, "
    "then a final line that is exactly '#### block' (it violates) or '#### allow' "
    f"(it does not). Write nothing before {THINK_OPEN}."
    f"\n\nExample:\n{THINK_OPEN} The input asks for a cookie recipe, which the policy "
    f"does not restrict. {THINK_CLOSE}\n#### allow"
)


def build_dataset():
    ds = build_guard_dataset()

    def fmt(ex):
        user = f"{GUARD_INSTRUCTION}\n\nPOLICY:\n{ex['policy']}\n\nINPUT:\n{ex['input']}"
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
