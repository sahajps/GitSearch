from openai import OpenAI
from google import genai
from google.genai import types
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import re

set_seed(42)

client_openai = OpenAI(api_key=open("../../Config/openai_key.txt", encoding="utf-8").read())
client_google = genai.Client(api_key=open("../../Config/google_key.txt", encoding="utf-8").read())

###############################################################################
def call_openai_api(prompt, model_name):
    try:
        response = client_openai.responses.create(
            model=model_name,
            input=prompt
        )
    except:
        time.sleep(60)
        response = client_openai.responses.create(
            model=model_name,
            input=prompt
        )

    return response

def generate_with_openai(model_type, prompts, tweetIds, CNs, Logs):
    if model_type == "gpt-5-nano":
        model_name = "gpt-5-nano-2025-08-07"
    for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                if pr == "NA":
                    CNs[id] = "NA"
                    Logs[id] = {"prompt": "NA", "response": "NA"}
                    continue

                resp = call_openai_api(pr, model_name)

                CNs[id] = resp.output[-1].content[0].text
                Logs[id] = {"prompt": pr, "response": str(resp)}

    return CNs, Logs

###############################################################################
def call_gemini_api(prompt, model_name):
    config_google = types.GenerateContentConfig(tools=[])
    try:
        response = client_google.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_google
        )
    except:
        time.sleep(60)
        response = client_google.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_google
        )

    return response

def generate_with_gemini(model_type, prompts, tweetIds, CNs, Logs):
    for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                if pr == "NA":
                    CNs[id] = "NA"
                    Logs[id] = {"prompt": "NA", "response": "NA"}
                    continue

                resp = call_gemini_api(pr, model_type)

                CNs[id] = resp.text
                Logs[id] = {"prompt": pr, "response": str(resp)}

    return CNs, Logs

###############################################################################
def generate_with_qwen3(model_path, model_type, prompts, tweetIds, CNs, Logs):
    model = AutoModelForCausalLM.from_pretrained(model_path+model_type, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_type, trust_remote_code=True)
    
    for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
        if id not in CNs:
            if pr == "NA":
                CNs[id] = "NA"
                Logs[id] = {"prompt": "NA", "response": "NA"}
                continue

            messages = [
                {"role": "user", "content": pr}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=8192, # Default as mentioned in Qwen3 docs: 32768
                do_sample=False
            )
            generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            resp = tokenizer.decode(generated_ids, skip_special_tokens=True)

            CNs[id] = resp.split("</think>")[-1].strip()
            Logs[id] = {"prompt": pr, "response": resp}
            torch.cuda.empty_cache()
            
    return CNs, Logs

###############################################################################
def generate_with_apriel_nemotron(model_path, model_type, prompts, tweetIds, CNs, Logs):
    model = AutoModelForCausalLM.from_pretrained(model_path+model_type, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_type, trust_remote_code=True)
    
    for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
        if id not in CNs:
            if pr == "NA":
                CNs[id] = "NA"
                Logs[id] = {"prompt": "NA", "response": "NA"}
                continue

            messages = [
                {"role": "user", "content": pr}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=[]
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Remove token_type_ids if present (fixing error observed during inference)
            if "token_type_ids" in model_inputs:
                model_inputs.pop("token_type_ids")

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=8192, # As per Nemotron docs, max context length is 65536 tokens
                do_sample=False
            )
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            resp = output.split("[BEGIN FINAL RESPONSE]")[-1].split("[END FINAL RESPONSE]")[0].strip()

            CNs[id] = resp
            Logs[id] = {"prompt": pr, "response": output}
            torch.cuda.empty_cache()
            
    return CNs, Logs

###############################################################################
def generate_with_open_source_model(model_path, model_type, prompts, tweetIds, CNs, Logs):
    model = AutoModelForCausalLM.from_pretrained(model_path+model_type, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_type, trust_remote_code=True)
    
    for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
        if id not in CNs:
            if pr == "NA":
                CNs[id] = "NA"
                Logs[id] = {"prompt": "NA", "response": "NA"}
                continue

            messages = [
                {"role": "user", "content": pr}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            CNs[id] = resp
            Logs[id] = {"prompt": pr, "response": resp}
            torch.cuda.empty_cache()
            
    return CNs, Logs