from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

model_path = "/home/Y1/finetuned_model"



import boto3
import os

# AWS credentials and region
aws_access_key_id = 'AKIAUL2NMFUUZDKIVIRB'
aws_secret_access_key = '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'
aws_region = 'eu-north-1'


def download_weight_from_s3():
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    # Bucket name and local directory path
    bucket_name = 'yi1'
    local_directory = '/home/Y1'

    # Folder path you want to download (e.g., 'folder_name/')
    folder_path = 'finetuned_model/'
    model_folder_path = local_directory +'/' + folder_path
    os.makedirs(model_folder_path, exist_ok=True)
    # List all objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']

    # Download objects from the specified folder
    for obj in objects:
        key = obj['Key']
        if key.startswith(folder_path):
            local_file_path = os.path.join(local_directory, key)
            print(local_file_path,"=========================================================")
            if not os.path.exists(local_file_path):
                s3.download_file(bucket_name, key, local_file_path)
                print(f'Downloaded {key} to {local_file_path}')
    return True

download_weight_from_s3()
# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#deepspeed.init_distributed()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

@app.post("/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt1 = data.get("prompt1", "")
    max_new_tokens = data.get("max_new_tokens", 100)
    min_new_tokens = data.get("min_new_tokens", -1)
    do_sample = data.get("do_sample", True)
    repetition_penalty = data.get("repetition_penalty", 1.3)
    no_repeat_ngram_size = data.get("no_repeat_ngram_size", 4)
    temperature = data.get("temperature", 0.95)
    top_k = data.get("top_k", 40)
    top_p = data.get("top_p", 0.95)
    #with open('data.txt', 'r') as file:
    #    prompt = file.read().replace('\n', '')

    prompt =  prompt1

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, max_length=1000, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                do_sample=do_sample, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                                temperature=temperature, top_k=top_k, top_p=top_p)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return {"response": response}



if __name__ == "__main__":
    download_weight_from_s3()
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=False, workers=0)

