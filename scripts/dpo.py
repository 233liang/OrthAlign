import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from dataclasses import dataclass, field
from typing import Optional
import pickle  # For saving and loading Python objects
import torch
import sys
import torch.nn as nn
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from typing import TYPE_CHECKING, Optional,Dict, List, Union, Any, Tuple
from src.trainer.dpo_trainer import DPOTrainer  
from src.utils import print_local_main, disable_progress_bar_non_local_main, param_sharding_enabled, set_seeds
U_BASIS_PATH = "your_u_basis_path_here"
V_BASIS_PATH = "your_v_basis_path_here"
global_null_space_bases: Dict[str, Dict[str, torch.Tensor]] = {
    "left_singular": {},
    "right_singular": {},
}

def load_bases(path: str, basis_type: str):
    """Generic loading function"""
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                global_null_space_bases[basis_type] = pickle.load(f)
            print(f"Successfully loaded {len(global_null_space_bases[basis_type])} precomputed {basis_type} singular vector basis matrices.")
        else:
            if int(os.environ.get("RANK", 0)) == 0:
                print(f"Error: {basis_type} singular vector basis file '{path}' not found. Please generate this file first.")
            sys.exit(1)
    except Exception as e:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Error occurred while loading {basis_type} singular vector basis matrices: {e}")
        sys.exit(1)

# Load basis matrices when file is imported
load_bases(U_BASIS_PATH, "left_singular")

def apply_null_space_projection_hooks(
    model: nn.Module, 
    left_singular_bases: Dict[str, torch.Tensor],
    right_singular_bases: Dict[str, torch.Tensor]
):
    """
    Iterate through model's LoRA parameters, register backward hooks for lora_B.weight and lora_A.weight parameters,
    project their gradients to corresponding left singular vector and right singular vector null spaces respectively.
    
    Args:
        model: Model that needs hook registration.
        left_singular_bases: Left singular vector basis matrix dictionary, key is module name, value is U matrix (d, r_u).
        right_singular_bases: Right singular vector basis matrix dictionary, key is module name, value is V matrix (d, r_v).
    """
    print("\nRegistering LoRA gradient null space bidirectional projection hooks...")
    registered_count = 0
    for name, param in model.named_parameters():
        if "A.default" in name:
           # param.requires_grad = False
           print("Debug output:", name)
        if not param.requires_grad:
            continue
            
        module_base_name = name.rsplit(".lora_B.default.weight", 1)[0] \
                         if ".lora_B.default.weight" in name else \
                         name.rsplit(".lora_A.default.weight", 1)[0] \
                         if ".lora_A.default.weight" in name else None
        
        if module_base_name is None:
            continue
            
        # Select corresponding basis matrix based on parameter name
        if ".lora_B.default.weight" in name:
            bases_dict = left_singular_bases
            basis_type = "left"
        elif ".lora_A.default.weight" in name:
            bases_dict = right_singular_bases
            basis_type = "right"
        else:
            continue

        if module_base_name in bases_dict:
            basis_for_this_layer = bases_dict[module_base_name].to(param.device).detach()
            basis_for_this_layer = basis_for_this_layer.to(dtype=torch.float16)
#            basis_for_this_layer = basis_for_this_layer.to(grade.dtype)
            # Define Hook function (using closure to capture basis matrix, parameter name and type)
            def hook_fn(grad, basis=basis_for_this_layer, param_name=name, basis_type=basis_type):
                if basis_type == "right":  # For right singular vectors (i.e., lora_A gradients)
                    # LoRA-A gradient shape is (r, d). V shape is (d, r_null).
                    # We can directly perform grad @ V_basis multiplication.
                    if grad.shape[1] == basis.shape[0]:
                        # Step-wise multiplication: (grad @ V_null) @ V_null.T
                        intermediate = torch.matmul(grad, basis) # (r, d) @ (d, r_null) -> (r, r_null)
                        projected_grad = torch.matmul(intermediate, basis.T) # (r, r_null) @ (r_null, d) -> (r, d)
                        return projected_grad
                    else:
                        print(f"警告 (Hook): 梯度维度与基矩阵不匹配，跳过投影: {param_name}, grad_shape={grad.shape}, basis_shape={basis.shape}")
                        return grad
                else:  # For left singular vectors (i.e., lora_B gradients)
                    # LoRA-B gradient shape is (d, r). U shape is (d, r_null).

                    if grad.shape[0] == basis.shape[0]:
                        intermediate = torch.matmul(basis.T, grad)
                        projected_grad = torch.matmul(basis, intermediate)
                        return projected_grad
                    else:
                        print(f"警告 (Hook): 梯度维度与基矩阵不匹配，跳过投影: {param_name}, grad_shape={grad.shape}, basis_shape={basis.shape}")
                        return grad
            
            # Register Hook
            param.register_hook(hook_fn)
            print(f"  ✓ Hook registered for {basis_type} singular vector projection on: {name}")
            registered_count += 1
        else:
            print(f"Warning: {basis_type} singular vector basis matrix for {module_base_name} not found, no hook registered for {name}.")

    print(f"LoRA gradient null space bidirectional projection hooks registration completed. Registered {registered_count} hooks in total.")
    return registered_count






disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft model name"})
    train_data_path: str = field(default='')
    eval_data_path: str = field(default='')
    use_flash_attention_2: Optional[bool] = field(default=True, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default='', metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(default=1700, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})
    generate_during_eval: Optional[bool] = field(default=True, metadata={"help": "whether to generate during evaluation"})
    output_dir: Optional[str] = field(default='')

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="",
            overwrite_output_dir=True,
            seed=42,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=8e-5,
            lr_scheduler_type="cosine",
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_modpo",
            report_to="none",
            num_train_epochs=3,
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="no",
            save_total_limit=0,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=False,
        )
    )

    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    )

script_args = tyro.cli(ScriptArguments)
seed=script_args.training_args.seed
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

################################# model/tokenizer #################################
print_local_main(f"loading model from {script_args.sft_model_name}...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=False,
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)

sft_model.config.update({
    "use_cache": False,
    "pad_token_id": sft_model.config.eos_token_id 
})




tokenizer = AutoTokenizer.from_pretrained('your_tokenizer_path_here')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

################################# dataset #################################
from datasets import load_dataset
from datasets import disable_caching
disable_caching()

train_dataset = load_dataset('json', data_files=script_args.train_data_path)['train']
def _saferlhf_to_preference_formatter(example):
    chat = [
        {"role": "user", "content": f"{example['raw_prompt']}"},
    ]
    return {
        "raw_prompt": example["raw_prompt"],
        "prompt":   tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
        "chosen":   example["chosen"],
        "rejected": example["rejected"],
    }

train_dataset = train_dataset.map(
    _saferlhf_to_preference_formatter,
    num_proc=6,
    remove_columns=train_dataset.column_names
)

eval_dataset = load_dataset('json', data_files=script_args.eval_data_path)['train']
eval_dataset = eval_dataset.map(
    _saferlhf_to_preference_formatter,
    num_proc=6,
    remove_columns=eval_dataset.column_names
)
################################# trainer #################################
print_local_main("Start training...")
print_local_main(f'Final checkpoint will be saved to: {script_args.training_args.output_dir}')

trainer = DPOTrainer(
    model=sft_model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=False,
    peft_config=script_args.peft_config,
)
apply_null_space_projection_hooks(
 trainer.model,  # Use trainer.model instead of sft_model
   global_null_space_bases["left_singular"],
 global_null_space_bases["right_singular"]
)
################################# save #################################
trainer.train()
save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))