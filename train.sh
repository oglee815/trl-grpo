#!/bin/bash
# GRPO launcher for the AI-safety-guard task (multi-GPU via torchrun).
#
#   bash train.sh --model /path/to/gemma-4-E2B-it --data dataset/guard_train.jsonl \
#                 --epochs 3 --gpus 8 --batch 1 --accum 8 --num_generations 16 --native
#
# Local toy (Qwen, 1 GPU, LoRA, <think> format):
#   bash train.sh --model Qwen/Qwen2.5-1.5B-Instruct --gpus 1 --num_generations 8 \
#                 --think_open "<think>" --think_close "</think>" --use_peft
set -euo pipefail

# ---- defaults ----
model=""
data=""                          # JSONL path; empty -> built-in toy set
epochs=3
lr=1e-5
gpus=8
batch=1
accum=8
num_generations=16               # MUST divide gpus*batch*accum
max_prompt_length=2048
max_completion_length=1024
max_think_tokens=512
brevity_weight=0.5
target_length=0                  # >0: correct answers rewarded for reasoning NEAR this length
length_tolerance=0               # tokens from target where that reward hits 0 (0 = max_think_tokens)
target_ratio=0                   # >0: target = target_ratio * policy tokens (scale with policy length)
tolerance_ratio=0                # >0: tolerance = tolerance_ratio * policy tokens
temperature=1.0
save_steps=50
max_steps=-1
native=false                     # Gemma native thought channel (<|think|>)
think_open="<|channel>thought"
think_close="<channel|>"
attn="flash_attention_2"
dtype="bfloat16"
use_peft=false
grad_ckpt=true                   # if GRPO completions come out empty/garbled, set false
dump=false                       # --dump: print one group of completions once (diagnostic)
dump_every=0                     # --dump_every N: print a group every N reward batches
dump_chars=0                     # --dump_chars N: chars per completion (0 = full text)
output_base="${OUTPUT_BASE:-./output}"

# a boolean switch may be given bare (--native) or with a value (--native true|false)
is_bool() { [[ "${1:-}" == "true" || "${1:-}" == "false" ]]; }

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --data) data="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --lr) lr="$2"; shift ;;
    --gpus) gpus="$2"; shift ;;
    --batch) batch="$2"; shift ;;
    --accum) accum="$2"; shift ;;
    --num_generations) num_generations="$2"; shift ;;
    --max_prompt_length) max_prompt_length="$2"; shift ;;
    --max_completion_length) max_completion_length="$2"; shift ;;
    --max_think_tokens) max_think_tokens="$2"; shift ;;
    --brevity_weight) brevity_weight="$2"; shift ;;
    --target_length) target_length="$2"; shift ;;
    --length_tolerance) length_tolerance="$2"; shift ;;
    --target_ratio) target_ratio="$2"; shift ;;
    --tolerance_ratio) tolerance_ratio="$2"; shift ;;
    --temperature) temperature="$2"; shift ;;
    --save_steps) save_steps="$2"; shift ;;
    --max_steps) max_steps="$2"; shift ;;
    --think_open) think_open="$2"; shift ;;
    --think_close) think_close="$2"; shift ;;
    --attn) attn="$2"; shift ;;
    --dtype) dtype="$2"; shift ;;
    --native)   if is_bool "${2:-}"; then native="$2";   shift; else native=true;   fi ;;
    --use_peft) if is_bool "${2:-}"; then use_peft="$2"; shift; else use_peft=true; fi ;;
    --no_grad_ckpt) grad_ckpt=false ;;
    --dump)     if is_bool "${2:-}"; then dump="$2";     shift; else dump=true;     fi ;;
    --dump_every) dump=true; dump_every="$2"; shift ;;
    --dump_chars) dump=true; dump_chars="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

[[ -z "$model" ]] && { echo "ERROR: --model is required"; exit 1; }

model_name=$(basename "$model")
data_name=$(basename "${data:-toy}")
output_dir="${output_base}/${model_name}/grpo_${lr}_${epochs}_${data_name}_${max_completion_length}"
mkdir -p "$output_dir"

extra=()
$native    && extra+=(--native_thinking)
$grad_ckpt && extra+=(--gradient_checkpointing)
$use_peft  && extra+=(--use_peft --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear)
$dump      && extra+=(--dump_completions --dump_every "${dump_every}" --dump_chars "${dump_chars}")

echo "output_dir=${output_dir}"
# --standalone: single-node job with an auto-picked FREE rendezvous port, so two
# concurrent runs on the same box (e.g. one per GPU) don't collide on port 29500.
torchrun --standalone --nproc_per_node="${gpus}" train_guard.py \
    --model_name_or_path="${model}" \
    --dataset_name="${data}" \
    --output_dir="${output_dir}" \
    --num_train_epochs="${epochs}" \
    --per_device_train_batch_size="${batch}" \
    --gradient_accumulation_steps="${accum}" \
    --learning_rate="${lr}" \
    --num_generations="${num_generations}" \
    --max_completion_length="${max_completion_length}" \
    --max_think_tokens="${max_think_tokens}" \
    --brevity_weight="${brevity_weight}" \
    --target_length="${target_length}" \
    --length_tolerance="${length_tolerance}" \
    --target_ratio="${target_ratio}" \
    --tolerance_ratio="${tolerance_ratio}" \
    --temperature="${temperature}" \
    --think_open="${think_open}" \
    --think_close="${think_close}" \
    --attn_implementation="${attn}" \
    --dtype="${dtype}" \
    --logging_steps=1 \
    --save_steps="${save_steps}" \
    --max_steps="${max_steps}" \
    --bf16=True \
    --report_to=tensorboard \
    "${extra[@]}" \
    > "${output_dir}/log.txt" 2>&1
