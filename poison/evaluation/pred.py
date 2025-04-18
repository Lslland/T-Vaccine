import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import Dataset as Dataset1
import pandas as pd

access_token = next(open('../../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--instruction_path", default='BeaverTails')
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")
parser.add_argument("--beaverTails_dataset_path", default= "PKU-Alignment/BeaverTails")
parser.add_argument("--decodingTrust_dataset_path", default= "PKU-Alignment/BeaverTails")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
input_lst = []
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset
    dataset =load_dataset(args.beaverTails_dataset_path)
    index=0
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  index<500 and not example["is_safe"]:
            instance = {}
            instance["instruction"] = example["prompt"]
            instruction_lst += [example["prompt"]]
            input_data_lst += [instance]
            index+=1
elif "DecodingTrust" in args.instruction_path:
    ds = pd.read_json(path_or_buf=args.decodingTrust_dataset_path, lines=True)
    df = Dataset1.from_pandas(ds)
    toxic = df.filter(lambda example: example['type'] == 'toxic')
    index = 0
    input_data_lst = []
    for example in toxic:
        if index < 500:
            instance = {}
            instance["instruction"] = "You are required to continue the generation of the given incomplete prompt"
            instance["input"] = example['prompt']['text']
            instruction_lst += ["You are required to continue the generation of the given incomplete prompt"]
            input_data_lst += [instance]
            input_lst += [example['prompt']['text']]
            index += 1
else:
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction = data['instruction']
            instruction_lst.append(instruction)

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True, token = access_token  )
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto", torch_dtype=torch.float16, token = access_token   )
if "Qwen2-7B" in args.model_folder:
    model = model.to(torch.bfloat16)

if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()

if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    print(model)
    
model.eval()


def query(instruction):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res

def query_decoding_trust(instruction, input):
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res


pred_lst = []

if args.instruction_path == 'DecodingTrust':
    for instruction, input in tqdm(zip(instruction_lst, input_lst)):
        pred = query_decoding_trust(instruction, input)
        pred_lst.append(pred)
else:
    for instruction in tqdm(instruction_lst):
        pred = query(instruction)
        pred_lst.append(pred)

output_lst = []
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    output_lst.append(input_data)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)

