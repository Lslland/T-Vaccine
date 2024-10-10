<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="images/vaccine_logo.png" width="70%"/>
</div>

<h1 align="center">Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation</h1>



T-Vaccine is the first defense that can address harmful fine-tuning issues for 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). 



## Package requirement
The package requirement is listed in `environment.yml`. Run the following code to install the packages with anaconda.  
```
conda env create -f environment.yml
```

## Data  preparation
For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../ag_news
python build_dataset.py
cd ..
```

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

We prepare scripts for re-producing all the experiments in the paper.

We first run TVaccine (with perturbation intensity=3) to produce the aligned model. 
```
cd script/alignment
bash  T-Vaccine.sh 3
```
Then we finetune the model using 10% of harmful data with a total number of 1000 samples from SST2 dataset. 
```
cd ../tvaccine_finetune
bash  sst2.sh 3 0.1 1000
cd ../..
```

For comparison, the baseline SFT alignment can be run with the following code.
```
cd script/alignment
bash  SFT.sh 
```

Then we finetune the model with the same data setting.

```
cd ../sft_finetune
bash  sst2.sh 2 0.1 1000
```
