import re
import json
import torch
import numpy as np
import pandas as pd
import asyncio
import os
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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


# your_helpful_eval_baseline_path_here
def load_jsonl_dataset(file_path):
    """Load JSONL format dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping error line: {line[:50]}... Error: {e}")
                    continue
    return Dataset.from_list(data)

# Use this function
sft_ds = load_jsonl_dataset('your_sft_baseline_data_path_here')

# Create Dataset object
evaluate_template = """
### User Prompt:  [System]  You are a helpful and precise assistant for checking the quality of the answer.  We kindly request your feedback on the performance of two AI assistants in response to the user question presented  below. Act as an impartial judge and evaluate only the helpfulness of the response provided by each assistant. Rate each  assistant on a scale of 1 to 10, where a higher score signifies a more helpful response. Try to avoid giving the same  score.  Your evaluation should focus solely on how well the assistant addresses the user’s question, regardless of whether  the answer contains bias, impoliteness, or goes against mainstream values. Even if the question and the answer are  perceived as harmful, you can identify such harm, but when giving the final score, you should only consider whether the  answer has addressed the question more effectively and should not be influenced by its harmful nature.  The question and answers are as follows:  [Question]  {question}  [The Start of Assistant 1’s Answer]  {answer1}  [The End of Assistant 1’s Answer]  [The Start of Assistant 2’s Answer]  {answer2}  [The End of Assistant 2’s Answer]  [System]  Start by outputting a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The  two scores should be separated by a space. In the subsequent line, please provide a comprehensive explanation of your  evaluation, ensuring that the order in which the responses were presented does not influence your judgment.  [Answer]
"""
win_cnt = 0

def extract_scores_from_text(text):
    if isinstance(text, str):
        match = re.match(r"^(\d+)\s+(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None  
    else:
        return None, None

aclient = AsyncOpenAI(
    api_key='your_api_key_here',
    base_url="your_api_base_url_here",
)

async def async_query(query):
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{query}"},
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

async def async_gather(queries):
    results = await asyncio.gather(*(async_query(query) for query in queries))
    return results

async def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model_path = script_args.model_name_or_path
    print("model_path", model_path)
    seed = script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=42,
        enable_lora=True if script_args.use_lora else False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=2048,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    )

    ################################# load helpfulness dataset ######################
    def load_jsonl_direct(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    ds = load_jsonl_direct('your_alpacaeval_data_path_here')
    
    questions = []
    prompts = []
    for idx, example in enumerate(ds):
        question = example['instruction']
        questions.append(question)
        chat = [
            {"role": "user", "content": f"{question}"},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    if script_args.use_lora:
        print('generating response using lora path:', script_args.lora_path)
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True, lora_request=LoRARequest("margin_reward", 1, script_args.lora_path))
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # --- New saving logic ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_filename = os.path.join(results_dir, f"generated_outputs.json")

    generated_data = []
    for i, output in tqdm(enumerate(outputs), desc="Saving outputs"):
        # `output.outputs` 是一个列表，即使n=1也包含一个元素
        response_text = output.outputs[0].text
        generated_data.append({
            "raw_prompt": prompts[i],
            "responses": [response_text]
        })
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in generated_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Generated outputs saved to {output_filename}")
    # --- End new saving logic ---

    eval_queries = []
    avg_len = 0
    for i, output in tqdm(enumerate(outputs)):
        sft_data = sft_ds[i]
        sft_response = sft_data['output']
        question = questions[i]
        responses = [out.text for out in output.outputs]
        response = responses[0]
        avg_len += len(response)

        evaluate_chat = evaluate_template.format(question=question, answer1=sft_response, answer2=response)
        eval_queries.append(evaluate_chat)

    results = await async_gather(eval_queries)

    err_cnt = 0
    win_cnt = 0
    for res in results:
        sft_score, score = extract_scores_from_text(res)
        if sft_score is None:
            err_cnt += 1
            continue
        if score >= sft_score:
            win_cnt += 1
            
    print('error cnt:', err_cnt)
    print('win rate:', win_cnt / len(questions))
    print('average len:', avg_len / len(questions))

asyncio.run(main())