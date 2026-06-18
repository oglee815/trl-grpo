"""Pre-flight: does MODEL_NAME actually emit a *closable* think block under our
prompt? (If not, brevity/format rewards will read 0 and GRPO has nothing to push.)
Usage:  $env:MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"; python check_format.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import THINK_OPEN, THINK_CLOSE, extract_thinking, count_think_tokens

MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
INSTR = (
    f"Solve the problem. Put ALL reasoning between {THINK_OPEN} and {THINK_CLOSE}, "
    f"keep it SHORT, then end with '#### <answer>'.\n\nExample:\n"
    f"{THINK_OPEN} 48 + 24 = 72 {THINK_CLOSE}\n#### 72"
)
TEMP = float(os.environ.get("TEMP", "0.9"))
MAX_NEW = int(os.environ.get("MAX_NEW", "200"))
# a multi-step GSM8K-style question (closer to real training difficulty)
Q = ("James writes a 3-page letter to 2 different friends twice a week. "
     "How many pages does he write in a year?")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to("cuda" if torch.cuda.is_available() else "cpu")

ids = tok.apply_chat_template(
    [{"role": "user", "content": INSTR + "\n\n" + Q}],
    add_generation_prompt=True, return_tensors="pt",
).to(model.device)

print(f"MODEL={MODEL}\n")
ok = 0
for i in range(4):
    out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=0.95)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    th = extract_thinking(text)
    n = count_think_tokens(th, tok) if th else 0
    ok += th is not None
    print(f"--- sample {i}: closable_think={th is not None}, think_tokens={n}")
    print(text[:400].strip(), "\n")
print(f"==> {ok}/4 samples produced a closable think block")
