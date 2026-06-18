"""Verify the Gemma 4 thinking-channel format LOCALLY (tokenizer only, no GPU,
no model weights). Pins down the EXACT delimiters for reward.set_think_delimiters()
so the brevity reward parses correctly on the B300 run.
Usage:  python inspect_gemma4.py
"""
import os
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
tok = AutoTokenizer.from_pretrained(MODEL)

print("=== special_tokens_map ===")
print(tok.special_tokens_map)

vocab = tok.get_vocab()
hits = sorted((v, t) for t, v in vocab.items()
              if any(k in t.lower() for k in ("channel", "think", "thought", "<|", "|>")))
print("\n=== channel/think-ish special tokens (id, token) ===")
for v, t in hits[:60]:
    print(f"{v:>8}  {t!r}")

print("\n=== rendered chat template (system carries <|think|>) ===")
msgs = [{"role": "system", "content": "<|think|>"}, {"role": "user", "content": "What is 2+2?"}]
try:
    print(repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
except Exception as e:
    print("apply_chat_template failed:", repr(e))
