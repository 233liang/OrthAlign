# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import os
import sys
import pickle  # Used to save and load Python objects
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Any, Tuple
from torch.optim import AdamW
from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer

U_BASIS_PATH = "your_safety_left_nullspace_bases_last16.pkl"

# Global dictionary for storing left singular vector basis matrices
global_null_space_bases: Dict[str, Dict[str, torch.Tensor]] = {
    "left_singular": {},
}

def load_bases(path: str, basis_type: str):
    """Universal loading function"""
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                global_null_space_bases[basis_type] = pickle.load(f)
            print(f"Successfully loaded {len(global_null_space_bases[basis_type])} pre-computed {basis_type} singular vector basis matrices.")
        else:
            if int(os.environ.get("RANK", 0)) == 0:
                print(f"Error: {basis_type} singular vector basis file '{path}' not found. Please generate it first.")
            sys.exit(1)
    except Exception as e:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Error occurred while loading {basis_type} singular vector basis matrices: {e}")
        sys.exit(1)

# Load basis matrices when the file is imported
load_bases(U_BASIS_PATH, "left_singular")


def apply_null_space_projection_hooks(
    model: nn.Module, 
    left_singular_bases: Dict[str, torch.Tensor]
):
    """
    Traverse the model's LoRA parameters and register backward hooks on lora_B.weight parameters.
    Project their gradients to the nullspace of the corresponding left singular vectors.
    
    Args:
        model: The model to register hooks on.
        left_singular_bases: Dictionary of left singular vector basis matrices, key is module name, value is U matrix (d, r_u).
    """
    print("\nRegistering LoRA gradient nullspace projection hooks...")
    registered_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        module_base_name = name.rsplit(".lora_B.default.weight", 1)[0] \
                         if ".lora_B.default.weight" in name else None
        
        if module_base_name is None:
            continue

        if module_base_name in left_singular_bases:
            basis_for_this_layer = left_singular_bases[module_base_name].to(param.device).detach()

            # Define hook function (using closure to capture basis matrix and parameter name)
            def hook_fn(grad, basis=basis_for_this_layer, param_name=name):
                # For left singular vectors (i.e., gradients of lora_B)
                # LoRA-B gradient shape is (d, r). U shape is (d, r_null).
                if grad.shape[0] == basis.shape[0]:
                    intermediate = torch.matmul(basis.T, grad)
                    projected_grad = torch.matmul(basis, intermediate)
                    return projected_grad
                else:
                    print(f"Warning (Hook): Gradient dimension mismatch with basis matrix, skipping projection: {param_name}, grad_shape={grad.shape}, basis_shape={basis.shape}")
                    return grad
            
            # Register hook
            param.register_hook(hook_fn)
            print(f"  ✓ Hook registered for left singular vector projection on: {name}")
            registered_count += 1
        else:
            print(f"Warning: left singular vector basis matrix for {module_base_name} not found, skipping hook registration for {name}.")

    print(f"LoRA gradient nullspace projection hook registration completed. Total {registered_count} hooks registered.")
    return registered_count




def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
   
    apply_null_space_projection_hooks(
        model, 
        global_null_space_bases["left_singular"]
    )

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "rewards/accuracies"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
