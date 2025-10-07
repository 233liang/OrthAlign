#!/bin/bash
#SBATCH --job-name=g2
#SBATCH --output=jupyter_logs/eval-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here

module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"


# HuggingFaceH4/mistral-7b-sft-beta
# your_mistral_checkpoint_path_here

base_model=your_base_model_path_here
data_path=your_data_path_here
my_world_size=2
output_dir=your_output_dir_here
use_lora=false
lora_path=your_lora_path_here

# CUDA_VISIBLE_DEVICES="0,1" python safety_eval.py --model_name_or_path ${base_model} --dataset_path ${data_path} --output_dir ${output_dir} --local_index 0 --my_world_size ${my_world_size}
# CUDA_VISIBLE_DEVICES="0,1" python help_eval.py --model_name_or_path ${base_model} --dataset_path ${data_path} --output_dir ${output_dir} --local_index 0 --my_world_size ${my_world_size}

CUDA_VISIBLE_DEVICES="0,1" python help_eval.py --model_name_or_path ${base_model} --dataset_path ${data_path} --output_dir ${output_dir} --local_index 0 --my_world_size ${my_world_size} --use_lora ${use_lora} --lora_path ${lora_path}
#CUDA_VISIBLE_DEVICES="1,3,5" python help_eval.py --model_name_or_path ${base_model} --dataset_path ${data_path} --output_dir ${output_dir} --local_index 0 --my_world_size ${my_world_size}