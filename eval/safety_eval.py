#!/usr/bin/env python
import json
import torch
import numpy as np
import pandas as pd
import gc
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [128001, 128009], metadata={"help": "the ids of the end of sentence tokens"})
    use_lora: Optional[bool] = field(
        default=False
    )
    lora_path: Optional[str] = field(
        default='your_lora_path_here'
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
torch.manual_seed(seed)
np.random.seed(seed)


# --- 1. Load and Run the Primary Model (using vLLM) ---
print("--- Loading primary model with vLLM ---")
llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
    enable_lora=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024,
    n=1,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
)

df = pd.read_csv(script_args.dataset_path)
prompts = []
questions = []

for question in df['goal']:
    questions.append(question)
    chat = [
        {"role": "user", "content": f"{question}"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)

print("--- Generating responses ---")
if script_args.use_lora:
    print('generating response using lora path:', script_args.lora_path)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True, lora_request=LoRARequest("margin_reward", 1, script_args.lora_path))
else:
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

# --- 2. Release GPU memory from vLLM ---
print("--- Generation complete. Releasing vLLM resources to free GPU memory ---")
del llm
torch.cuda.empty_cache()
gc.collect()
print("--- GPU memory released. ---")

# --- 3. Load and Run the Llama-Guard Model ---
print("\n--- Loading Llama-Guard-3-8B for moderation ---")
model_id = "your_llama_guard_model_path_here"
device = "cuda:0"
dtype = torch.bfloat16

guard_tokenizer = AutoTokenizer.from_pretrained(model_id)
guard_llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = guard_llm.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

harmless_cnt = 0

print("--- Starting moderation check ---")
for i, output in enumerate(outputs):
    question = questions[i]
    responses = [out.text for out in output.outputs]
    response = responses[0]
    
    eval = moderate([
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ])

    if 'unsafe' not in eval:
        harmless_cnt += 1
    

print('')
print('*********dataset name:', script_args.dataset_path)
print('*********harmless rate:', harmless_cnt / len(questions))
print('')