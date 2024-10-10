#!/bin/bash
#SBATCH -J mesfa                 # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 240                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o agnews-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

RHO=$1
# density=$2
poison_ratio=$2
sample_num=$3
epoch=20
model_path=${4:-/models/meta-llama/Llama-2-7b-hf}
alignment_dataset_path=${5:-/data/beavertails_with_refusals_train.json} # alignment dataset
eval_model_path=${6:-/models/PKU-Alignment/beaver-dam-7b}
beaverTails_dataset_path=${7:-/data/PKU-Alignment/BeaverTails} # fine-tuning dataset
ag_news_path=${8:-/data/fancyzhx/ag_news}
lisa_activated_layers=8
lisa_interval_steps=20
prompt_data_size=200
probability_steps=200
path_after_slash=$(basename "$model_path") 
echo "The value of RHO is: $RHO"
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
cd  ../../                            # Change to working directory





CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--lora_folder ckpt/${path_after_slash}_TVaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--num_train_epochs 20 \
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
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 1000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/ag_news.json \
	--system_evaluate True \
	--lisa_activated_layers ${lisa_activated_layers} \
  --lisa_interval_steps ${lisa_interval_steps} \
  --prompt_data_size ${prompt_data_size} \
  --probability_steps ${probability_steps} \
	--alignment_dataset_path ${alignment_dataset_path} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--max_length 200 \


cd poison/evaluation  


# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_mesfa_${RHO} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/pred/mesfa_${RHO}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/pred/mesfa_${RHO}



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_TVaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--lora_folder2 ../../ckpt/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--eval_model_path ${eval_model_path}


cd ../../ag_news



CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_TVaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch}   \
	--lora_folder2 ../ckpt/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch}  \
	--model_folder ${model_path} \
	--output_path ../data/ag_news/${path_after_slash}_TVaccine_f_${RHO}_${poison_ratio}_${sample_num}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--ag_news_path ${ag_news_path}


wait
