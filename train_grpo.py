"""
Toy GRPO run (TRL) for length-aware reasoning rewards.

Local Windows toy:   python train_grpo.py
B300 (Linux) later:  set MODEL_NAME to Gemma 4, flip USE_VLLM=True, raise the
                     batch/steps/length knobs, and fix the THINK_* delimiters in
                     reward.py to Gemma's real thought-channel tokens.
"""
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import build_reward_funcs, THINK_OPEN, THINK_CLOSE

# --- Config (toy defaults) -------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-270m-it")  # smallest mobile Gemma
#   real run (B300): MODEL_NAME = "google/gemma-4-E2B-it" (needs transformers 5.x), and:
#     1) reward.set_think_delimiters("<|channel>thought", "<channel|>")  # VERIFIED tokens
#     2) enable thinking by pre-rendering prompts with
#        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
#                                      enable_thinking=True)  # injects <|think|> in system turn
#        and feed the rendered string as a (non-conversational) "prompt" column.
MAX_THINK_TOKENS = int(os.environ.get("MAX_THINK", "512"))  # brevity normalizer; lower = steeper length pressure
USE_VLLM = False          # vLLM is unreliable on Windows; keep off for the toy
N_TRAIN = 64              # tiny slice just to see the loop run
USE_LORA = os.environ.get("USE_LORA", "0") == "1"   # LoRA: fits >1B models on a 12GB GPU
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
NUM_GEN = int(os.environ.get("NUM_GEN", "4"))        # completions per prompt (GRPO group size)
LR = float(os.environ.get("LR", "1e-6"))
TEMP = float(os.environ.get("TEMP", "1.0"))          # generation sampling temperature
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "256"))  # raise so the think block can close
BREVITY_W = float(os.environ.get("BREVITY_W", "0.5"))  # brevity weight; MUST stay < 1.0 (correctness)
GRAD_CKPT = os.environ.get("GRAD_CKPT", "0") == "1"  # OFF by default: in this trl 0.19 +
#   transformers 4.57 stack, GC leaks into the generation phase (use_cache=False) and
#   corrupts completions. On the matched B300 stack (trl 1.5 + torch 2.5) you can re-enable it.

SYSTEM_PROMPT = (
    "You are a math solver. Your reply MUST start with the token "
    f"{THINK_OPEN}, then your step-by-step reasoning, then {THINK_CLOSE}, then the "
    f"final answer on a line '#### <number>'. Write nothing before {THINK_OPEN}."
    f"\n\nExample:\n{THINK_OPEN} There are 48 clips sold in April and half as many, "
    f"24, in May. Adding them: 48 + 24 = 72. {THINK_CLOSE}\n#### 72"
)


def build_dataset(n: int = N_TRAIN):
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n}]")

    def fmt(ex):
        # Gemma chat templates have no `system` role -> fold the instruction into
        # the user turn (behaviorally equivalent, avoids a template error).
        return {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + ex["question"]},
            ],
            "answer": ex["answer"],  # GSM8K gold ends with '#### <number>'
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    correctness, brevity, fmt = build_reward_funcs(
        tokenizer=tokenizer,
        max_think_tokens=MAX_THINK_TOKENS,
        wrong_answer_length_mode="shorter_better",  # the requested spec
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_init_kwargs = {"attn_implementation": "eager"}
    if USE_LORA:
        model_init_kwargs["torch_dtype"] = "bfloat16"   # load the frozen base in bf16

    peft_config = None
    if USE_LORA:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    reward_funcs = [correctness, brevity, fmt]
    reward_weights = [1.0, BREVITY_W, 0.2]   # w_c > w_b is the only hard rule
    if os.environ.get("DEBUG") == "1":
        from reward import get_completion_text
        _seen = []

        def debug_dump(prompts=None, completions=None, **kwargs):
            if not _seen:
                txt = get_completion_text(completions[0])
                print("\n===== DEBUG completion[0] repr (first 2500 chars) =====")
                print(repr(txt[:2500]))
                print("===== char_len:", len(txt), "=====\n", flush=True)
                _seen.append(1)
            return [0.0] * len(completions)

        reward_funcs.append(debug_dump)
        reward_weights.append(0.0)

    args = GRPOConfig(
        output_dir="out-grpo-toy",
        per_device_train_batch_size=NUM_GEN,
        gradient_accumulation_steps=1,
        num_generations=NUM_GEN,      # group size G (must divide the global batch)
        max_prompt_length=256,
        max_completion_length=MAX_COMPLETION,
        temperature=TEMP,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="no",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=USE_LORA and GRAD_CKPT,         # save activation memory on 12GB
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
    print("toy run finished -> ./out-grpo-toy")


if __name__ == "__main__":
    main()
