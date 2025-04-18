#!/bin/bash
#SBATCH -J vaccine                 # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o Vaccine_0-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

RHO=$1
epoch=20
model_path=${2:-/models/meta-llama/Llama-2-7b-hf}
alignment_dataset_path=${3:-/data/beavertails_with_refusals_train.json} # alignment dataset
eval_model_path=${4:-/models/PKU-Alignment/beaver-dam-7b}
beaverTails_dataset_path=${5:-/data/PKU-Alignment/BeaverTails} # fine-tuning dataset
path_after_slash=$(basename "$model_path") 
echo "The value of RHO is: $RHO"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_safe_alignment \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_vaccine_${RHO}_${epoch} \
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
	--optimizer vaccine \
	--rho $RHO \
	--system_evaluate True \
	--alignment_dataset_path ${alignment_dataset_path} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--max_length 200 \

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_vaccine_${RHO}_${epoch} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/${path_after_slash}_vaccine_${RHO}_${epoch} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_vaccine_${RHO}_${epoch} \
	--eval_model_path ${eval_model_path} \
