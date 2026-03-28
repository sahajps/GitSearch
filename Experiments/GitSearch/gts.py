import pandas as pd
import json
from prompt_temp import prompt_gap_identification, prompt_targeted_search_article, prompt_synthesize_final_note, timeToDate
from model_inference import generate_with_openai, generate_with_google, add_citations
from tqdm import tqdm
import sys
import os

os.makedirs("../Outputs", exist_ok=True)
os.makedirs("Logs", exist_ok=True)

model_type = sys.argv[1] # model_name: eg "gpt-5-nano", "gemini-2.5-flash"
run_time = sys.argv[2] # "T1", "T2", ...


# Load tweets & notes data
df_notes = pd.read_csv("../../Data/notes.tsv", sep="\t", low_memory=False)

dic_tweets = json.load(open("../../Data/tweets.json", encoding="utf-8"))

tweetIds = json.load(open("../../Data/test_helpful.json", encoding="utf-8"))["tweetId"]

# Notes context building
dic_notes = {}

for id in tweetIds:
    tmp = df_notes[df_notes["tweetId"]==int(id)]
    tmp = tmp[tmp["helpfulnessStatus"]=="NEEDS_MORE_RATINGS"]

    if(len(tmp)==0):
        # No notes with NEEDS_MORE_RATINGS to create supernote from.
        dic_notes[id] = "NA"
        continue

    # Getting (note, createdAt, helpfulnessScore) tuples
    # list(zip( ['a','b','c'], ['p','q','r'], ['x','y','z'])) --> [('a', 'p', 'x'), ('b', 'q', 'y'), ('c', 'r', 'z')]
    tmp_notes = list(zip(tmp["summary"].tolist(), tmp["createdAtMillis"].tolist()))
    tmp_notes = "\n".join([
      f"Note {idx+1} (Published on {timeToDate(createAt)}): {note}" for idx, (note, createAt) in enumerate(tmp_notes)
    ])

    dic_notes[id] = tmp_notes

# Main output processing function
def generate_with_model(prompt, model_name, use_web_search):
    if model_name=="gpt-5-nano":
        resp = generate_with_openai(prompt, use_web_search=use_web_search)

        return resp, resp.output[-1].content[0].text
    
    elif model_name=="gemini-2.5-flash":
        resp = generate_with_google(prompt, use_web_search=use_web_search)

        try:
            return resp, add_citations(resp)
        except:
            return resp, resp.text

# Inference     
try:
    CNs = json.load(open(f"../Outputs/our_{model_type}_{run_time}.json", encoding="utf-8"))
    Logs = json.load(open(f"Logs/logs_{model_type}_{run_time}.json", encoding="utf-8"))
    OLogs = json.load(open(f"Logs/logs_output_{model_type}_{run_time}.json", encoding="utf-8"))
except:
    CNs = {}
    Logs = {}
    OLogs = {}

for id in tqdm(tweetIds):
    if id in CNs:
        continue

    log_dic, log_out_dic = {}, {} # For logging
    gap_prompt = prompt_gap_identification(dic_tweets[id]["text"], dic_tweets[id]["createdAt"], dic_notes[id])
    log_dic["gap_prompt"], log_out_dic["gap_prompt"] = gap_prompt, gap_prompt

    gap_output = generate_with_model(gap_prompt, model_name=model_type, use_web_search=False)
    log_dic["gap_output"], log_out_dic["gap_output"] = str(gap_output[0]), gap_output[1] 

    targeted_search_article_prompt = prompt_targeted_search_article(dic_tweets[id]["text"], dic_tweets[id]["createdAt"], dic_notes[id], log_out_dic["gap_output"])
    log_dic["targeted_search_article_prompt"], log_out_dic["targeted_search_article_prompt"] = targeted_search_article_prompt, targeted_search_article_prompt

    targeted_search_article_output = generate_with_model(targeted_search_article_prompt, model_name=model_type, use_web_search=True)
    log_dic["targeted_search_article_output"], log_out_dic["targeted_search_article_output"] = str(targeted_search_article_output[0]), targeted_search_article_output[1]

    cn_synthesis_prompt = prompt_synthesize_final_note(dic_tweets[id]["text"], dic_tweets[id]["createdAt"], log_out_dic["targeted_search_article_output"])
    log_dic["cn_synthesis_prompt"], log_out_dic["cn_synthesis_prompt"] = cn_synthesis_prompt, cn_synthesis_prompt

    cn_synthesis_output = generate_with_model(cn_synthesis_prompt, model_name=model_type, use_web_search=False)
    log_dic["cn_synthesis_output"], log_out_dic["cn_synthesis_output"] = str(cn_synthesis_output[0]), cn_synthesis_output[1]

    Logs[id] = log_dic
    OLogs[id] = log_out_dic
    CNs[id] = cn_synthesis_output[1]

    # Saving results after each step
    json.dump(CNs, open(f"../Outputs/our_{model_type}_{run_time}.json", "w", encoding="utf-8"), indent=4)
    json.dump(Logs, open(f"Logs/logs_{model_type}_{run_time}.json", "w", encoding="utf-8"), indent=4)
    json.dump(OLogs, open(f"Logs/logs_output_{model_type}_{run_time}.json", "w", encoding="utf-8"), indent=4)