"""
Optimized version: Better utilize batch processing to improve efficiency
"""

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

# --- Configuration Parameters ---
LORA_BASE_DIR = "your_lora_base_dir_here"
REFERENCE_MODEL_PATH = "your_reference_model_path_here"
DATASET_PATH = "your_dataset_path_here"
NUM_SAMPLES = 3192
BATCH_SIZE = 512
BETA = 0.1
MAX_MODELS_TO_EVAL = 10

def get_log_probs_batch(model, input_ids_list, labels_list):
    """
    Batch compute log probabilities for efficiency
    """
    all_logps = []
    
    with torch.no_grad():
        for input_ids, labels in zip(input_ids_list, labels_list):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits.to(torch.float32)
            
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tensors
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            # Filter valid labels
            valid_indices = (flat_labels != -100).nonzero().flatten()
            
            if valid_indices.numel() == 0:
                logps = torch.zeros(input_ids.shape[0]).to(input_ids.device)
            else:
                valid_flat_logits = flat_logits.index_select(0, valid_indices)
                valid_flat_labels = flat_labels.index_select(0, valid_indices)
                
                log_probs = F.log_softmax(valid_flat_logits, dim=-1)
                logps_per_token = torch.gather(log_probs, dim=1, index=valid_flat_labels.unsqueeze(1)).squeeze(1)

                # Re-insert log probabilities
                full_logps = torch.zeros_like(flat_labels, dtype=torch.float32)
                full_logps.index_copy_(0, valid_indices, logps_per_token)
                
                # Reshape and sum
                full_logps = full_logps.view(shift_labels.shape)
                logps = full_logps.sum(dim=-1)
            
            all_logps.append(logps)
    
    return all_logps

def prepare_batch_data(data_batch, tokenizer, device):
    """
    Prepare batch data, return chosen and rejected data separately
    """
    chosen_input_ids = []
    chosen_labels = []
    rejected_input_ids = []
    rejected_labels = []
    
    for item in data_batch:
        prompt = item["prompt"]
        chosen_response = item["chosen"]
        rejected_response = item["rejected"]
        
        # Tokenize
        chosen_inputs = tokenizer(prompt + chosen_response, return_tensors="pt", padding=True, truncation=True)
        rejected_inputs = tokenizer(prompt + rejected_response, return_tensors="pt", padding=True, truncation=True)
        
        # Prepare labels
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        
        chosen_labels_tensor = chosen_inputs["input_ids"].clone()
        rejected_labels_tensor = rejected_inputs["input_ids"].clone()
        
        chosen_labels_tensor[:, :prompt_len] = -100
        rejected_labels_tensor[:, :prompt_len] = -100
        
        # Move to device and add to lists
        chosen_input_ids.append(chosen_inputs["input_ids"].to(device))
        chosen_labels.append(chosen_labels_tensor.to(device))
        rejected_input_ids.append(rejected_inputs["input_ids"].to(device))
        rejected_labels.append(rejected_labels_tensor.to(device))
    
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

def calculate_batch_accuracy(chosen_logps_list, rejected_logps_list, ref_chosen_logps_list, ref_rejected_logps_list, beta, device):
    """
    Batch calculate accuracy
    """
    accuracies = []
    
    for i in range(len(chosen_logps_list)):
        # Calculate rewards
        chosen_rewards = beta * (chosen_logps_list[i] - ref_chosen_logps_list[i].to(device))
        rejected_rewards = beta * (rejected_logps_list[i] - ref_rejected_logps_list[i].to(device))
        
        # Calculate accuracy for this sample
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()  # Use mean() to handle possible batch dimension
        accuracies.append(accuracy)
    
    return accuracies

def main():
    # Device setup
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 GPUs.")
    
    ref_device = torch.device("cuda:0")
    policy_device = torch.device("cuda:1")
    
    print(f"Reference model: {ref_device}")
    print(f"Policy models: {policy_device}")
    print("-" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    sampled_data = random.sample(list(dataset), NUM_SAMPLES)
    
    # Pre-compute reference model outputs
    print("\nPre-computing reference model outputs...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        REFERENCE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=ref_device
    )
    ref_model.eval()

    precomputed_ref_outputs = []
    
    for i in tqdm(range(0, NUM_SAMPLES, BATCH_SIZE), desc="Reference model"):
        batch = sampled_data[i:i+BATCH_SIZE]
        
        # Prepare batch data
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = prepare_batch_data(batch, tokenizer, ref_device)
        
        # Compute log probabilities
        chosen_logps = get_log_probs_batch(ref_model, chosen_ids, chosen_labels)
        rejected_logps = get_log_probs_batch(ref_model, rejected_ids, rejected_labels)
        
        precomputed_ref_outputs.append((chosen_logps, rejected_logps))
    
    # Clean up reference model
    del ref_model
    torch.cuda.empty_cache()

    # Evaluate LoRA models
    print("\nEvaluating LoRA models...")
    results = {}
    
    lora_folders = [f for f in os.listdir(LORA_BASE_DIR) if f.startswith("noise_k")]
    lora_folders.sort(key=lambda x: int(x.replace("noise_k", "")))
    lora_folders_to_eval = lora_folders[1:11]

    for folder in tqdm(lora_folders_to_eval, desc="LoRA models"):
        try:
            lora_path = os.path.join(LORA_BASE_DIR, folder)
            
            # Load model
            base_model = AutoModelForCausalLM.from_pretrained(
                REFERENCE_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map=policy_device
            )
            policy_model = PeftModel.from_pretrained(base_model, lora_path)
            policy_model.eval()
            
            # Get rank
            with open(os.path.join(lora_path, "adapter_config.json"), 'r') as f:
                current_rank = json.load(f).get("r", "N/A")
            
            all_accuracies = []
            
            # Process each batch
            for i in range(0, NUM_SAMPLES, BATCH_SIZE):
                batch = sampled_data[i:i+BATCH_SIZE]
                batch_idx = i // BATCH_SIZE
                
                # Prepare batch data for policy model
                chosen_ids, chosen_labels, rejected_ids, rejected_labels = prepare_batch_data(batch, tokenizer, policy_device)
                
                # Get policy model outputs
                policy_chosen_logps = get_log_probs_batch(policy_model, chosen_ids, chosen_labels)
                policy_rejected_logps = get_log_probs_batch(policy_model, rejected_ids, rejected_labels)
                
                # Get reference outputs
                ref_chosen_logps, ref_rejected_logps = precomputed_ref_outputs[batch_idx]
                
                # Calculate batch accuracy
                batch_accuracies = calculate_batch_accuracy(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    BETA, policy_device
                )
                
                all_accuracies.extend(batch_accuracies)
                # print("Debug output:", batch_accuracies)
            avg_accuracy = np.mean(all_accuracies)
            results[current_rank] = avg_accuracy
            
            print(f"✅ {folder} (rank={current_rank}): {avg_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error with {folder}: {e}")
        finally:
            try:
                del policy_model, base_model
            except:
                pass
            torch.cuda.empty_cache()

    # Results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    for rank, accuracy in sorted(results.items()):
        print(f"Rank {rank:>3}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
