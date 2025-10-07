#!/bin/bash
# module avail
module load cuda/12.5.1
module load gcc/11.3.0

# ss -tulnp | grep 29500

LAUNCH="$(which python) -m accelerate.commands.launch --config_file configs/zero_2.yaml --num_processes=4"

rm_run_name=Orthalign-ultra
sft_model_name=your_sft_model_path_here
train_data_path=your_train_data_path_here
output_dir=your_output_dir_here

eval_data_path=your_eval_data_path_here
prompt_template="<|user|>\n{raw_prompt}</s>\n<|assistant|>\n" # llama / mistral 
learning_rate=1e-4

echo "train_data_path: $train_data_path"
echo "rm_run_name: $rm_run_name"

CUDA_VISIBLE_DEVICES="0,1,2,3" PYTHONPATH=. $LAUNCH dpo.py \
    --sft_model_name ${sft_model_name} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --beta 0.1 \
    --prompt_template ${prompt_template} \
    --training_args.run_name ${rm_run_name} \
    --training_args.learning_rate ${learning_rate} \
    --training_args.output_dir ${output_dir}/${rm_run_name} \
    --training_args.num_train_epochs 2 \
    --training_args.load_best_model_at_end False

echo "save model to ${output_dir}/${rm_run_name}"
