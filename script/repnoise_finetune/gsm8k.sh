#!/bin/bash
#SBATCH -J vaccine                 # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o gsm8k-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

alpha=0.1
beta=0.001
bad_sample_num=2000
# density=$2
poison_ratio=$1
sample_num=$2
model_path=${3:-/models/meta-llama/Llama-2-7b-hf}
alignment_dataset_path=${4:-/data/beavertails_with_refusals_train.json} # alignment dataset
eval_model_path=${5:-/models/PKU-Alignment/beaver-dam-7b}
beaverTails_dataset_path=${6:-/data/PKU-Alignment/BeaverTails} # fine-tuning dataset
gsm8k_path=${7:-/data/openai/gsm8k}
path_after_slash=$(basename "$model_path") 
echo "The value of RHO is: $RHO"
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory





CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--lora_folder ckpt/${path_after_slash}_repnoise_${alpha}_${beta}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num} \
	--num_train_epochs 50 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 100 \
	--tf32 True \
	--eval_steps 1000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--system_evaluate True \
	--rho ${alpha}  \
	--lamb ${beta} \
	--bad_sample_num ${bad_sample_num} \
	--alignment_dataset_path ${alignment_dataset_path} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--max_length 100 \


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_vaccine_${RHO} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/pred/vaccine_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/vaccine_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num} \
	--eval_model_path ${eval_model_path}



cd ../../gsm8k


CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_repnoise_${alpha}_${beta}   \
	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num}  \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_${poison_ratio}_${sample_num} \
	--gsm8k_path ${gsm8k_path}


wait
