#!/bin/bash
#SBATCH -J vaccine                 # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o Vaccine_0-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

alpha=0.1
beta=0.001
bad_sample_num=2000
epoch=20

model_path=${1:-/models/Qwen/Qwen2-7B}
alignment_dataset_path=${2:-/data/beavertails_with_refusals_train.json} # alignment dataset
eval_model_path=${3:-/models/PKU-Alignment/beaver-dam-7b}
decodingTrust_dataset_path=${4:-/data/decoding_trust/training_dataset.jsonl}

path_after_slash=$(basename "$model_path") 
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=1  python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_safe_alignment \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_${epoch} \
	--num_train_epochs ${epoch} \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-3 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer rep_noise \
	--rho ${alpha}  \
	--lamb ${beta} \
	--bad_sample_num ${bad_sample_num} \
	--system_evaluate True \
	--alignment_dataset_path ${alignment_dataset_path} \
	--decodingTrust_dataset_path ${decodingTrust_dataset_path} \
	--max_length 100 \
	--harmful_dataset "DecodingTrust" \

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=1  python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_${epoch} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}_${epoch} \
	--decodingTrust_dataset_path ${decodingTrust_dataset_path} \
	--instruction_path "DecodingTrust"

CUDA_VISIBLE_DEVICES=1  python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}_${epoch} \
	--eval_model_path ${eval_model_path} \
