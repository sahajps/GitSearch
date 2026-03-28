import pandas as pd
import json
from prompt_temp import prompt_for_supernote_lite
from model_inference import generate_with_openai, generate_with_gemini, generate_with_qwen3, generate_with_apriel_nemotron, generate_with_open_source_model
from tqdm import tqdm
import sys
import os
import traceback

os.makedirs("../Outputs", exist_ok=True)
os.makedirs("Logs", exist_ok=True)

model_type = sys.argv[1] # model_name: eg "gpt-5-nano", Qwen3-14B etc.
model_path = sys.argv[2] # path to open models (hf or local) otherwise "NA": eg. "/home/sahaj/Models/", "meta-llama/" etc.
if model_path!="NA":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

# Load tweets & notes data
df_notes = pd.read_csv("../../Data/notes.tsv", sep="\t", low_memory=False)

# Adding helpfulnessScore column
df_notes["helpfulnessScore"] = df_notes["helpfulRatings"] / df_notes["totalRatings"]

dic_tweets = json.load(open("../../Data/tweets.json", encoding="utf-8"))

test_help = json.load(open("../../Data/test_helpful.json", encoding="utf-8"))

tweetIds = test_help["tweetId"]

# Prompt building
prompts = []

for id in tweetIds:
    txt = dic_tweets[id]["text"]
    date = dic_tweets[id]["createdAt"]

    tmp = df_notes[df_notes["tweetId"]==int(id)]
    tmp = tmp[tmp["helpfulnessStatus"]=="NEEDS_MORE_RATINGS"]

    if(len(tmp)==0):
        # No notes with NEEDS_MORE_RATINGS to create supernote from.
        prompts.append("NA")
        continue

    # Getting (note, createdAt, helpfulnessScore) tuples
    # list(zip( ['a','b','c'], ['p','q','r'], ['x','y','z'])) --> [('a', 'p', 'x'), ('b', 'q', 'y'), ('c', 'r', 'z')]
    comm_notes = list(zip(tmp["summary"].tolist(), tmp["createdAtMillis"].tolist(), tmp["helpfulnessScore"].tolist()))

    prompts.append( prompt_for_supernote_lite(txt, date, comm_notes) )

# Calling model to write Super Community Notes for the above tweets...
try:
    CNs = json.load(open(f"../Outputs/supernote_lite_{model_type}.json", encoding="utf-8"))
    Logs = json.load(open(f"Logs/logs_{model_type}.json", encoding="utf-8"))
except:
    CNs = {}
    Logs = {}

try:
    if model_type == "gpt-5-nano":
        CNs, Logs = generate_with_openai(model_type, prompts, tweetIds, CNs, Logs)
    elif model_type == "gemini-2.5-flash":
        CNs, Logs = generate_with_gemini(model_type, prompts, tweetIds, CNs, Logs)
    elif model_type == "Qwen3-14B":
        CNs, Logs = generate_with_qwen3(model_path, model_type, prompts, tweetIds, CNs, Logs)
    elif model_type == "Apriel-Nemotron-15b-Thinker":
        CNs, Logs = generate_with_apriel_nemotron(model_path, model_type, prompts, tweetIds, CNs, Logs)
    else:
        CNs, Logs = generate_with_open_source_model(model_path, model_type, prompts, tweetIds, CNs, Logs)
except:
    print("Error occurred. Saving progress so far...")
    traceback.print_exc()

# Save results
json.dump(CNs, open(f"../Outputs/supernote_lite_{model_type}.json", "w", encoding="utf-8"), indent=4)
json.dump(Logs, open(f"Logs/logs_{model_type}.json", "w", encoding="utf-8"), indent=4)

# Sample prompt:-
# You are an expert fact-checker. X (Twitter) has a crowd-sourced fact-checking program, called Community Notes. Here, users can write 'notes' on potentially misleading tweets. Each note needs to be rated helpful by a sufficient number of diversely-opinionated people (note-raters) for it to be shown publicly alongside the piece of content.

# Helpful attributes in notes include:
# - Cites high-quality sources
# - Easy to understand
# - Directly addresses the post's claim
# - Provides important context
# - Neutral or unbiased language

# Unhelpful attributes in notes include:
# - Sources not included or unreliable
# - Sources do not support note
# - Incorrect information
# - Opinion or speculation
# - Typos or unclear language
# - Misses key points or irrelevant
# - Argumentative or biased language
# - Note not needed on this post
# - Harassment or abuse

# ### Task
# You will be given a potentially misleading tweet, its existing community notes, and the helpfulness scores (0-1) of those notes. Your task is to create a single helpful **Super Community Note** that balances the perspectives reflected in the helpfulness scores while summarizing the key points from the notes. The Supernote should provide clear, factual context that addresses the potentially misleading information in the tweet and be strong enough to replace all existing notes.
# - The note must be within 280 characters. Treat each URL as 1 character, regardless of its actual length.
# - The note must include one or more URLs to credible sources. Use only URLs explicitly provided in context. Do not invent or substitute other URLs.
# - The note must be neutral, factual, and concise. When possible, cite sources across the political spectrum to strengthen neutrality, but prioritize reliability and relevance.
# - Output only the Super Community Note text, with URLs included. Do not add explanations, formatting, or extra commentary.
# - Do not include any information beyond what is explicitly provided in the context.

# ---
# **Tweet (published on Mon Jul 14 04:19:58 +0000 2025):**
# """Trump refused to leave even after they were practically pushing him away so Chelsea photoshopped him out of their official trophy raising photo. Good move. https://t.co/igcVWyuVek"""

# **Existing Community Note(s):**
# """1. Chelsea FC has better graphic designers than this. Altered yes, but official, no.    https://x.com/chelseafc/status/1944522524848763278?s=46&amp;t=NLJX_nxjJrwre4jK3CicHA    https://x.com/tosinadarabioyo/status/1944791377050567110?s=46&amp;t=NLJX_nxjJrwre4jK3CicHA | Published on Mon Jul 14 21:43:45 +0000 2025; Helpfulness Score: 0.3465346534653465
# 2. https://www.instagram.com/p/DMEITEvMoLi/?igsh=MTVtZGswMTVkd3Nkag==    It depends on which official account you look at as the Chelsea FC official Instagram post doesn’t include Trump. Therefore NNN | Published on Tue Jul 15 00:10:31 +0000 2025; Helpfulness Score: 0.6075949367088608
# 3. Trump is in the photo on the official Chelsea account.  This photo without Trump was not released by Chelsea, it is also not photoshopped or its tools, the image is regenerated with AI (Look at the text on the trophy).    https://x.com/ChelseaFC/status/1944522524848763278    https://x.com/tosinadarabioyo/status/1944791377050567110    https://en.wikipedia.org/wiki/Generative_artificial_intelligence   | Published on Mon Jul 14 23:40:21 +0000 2025; Helpfulness Score: 0.6318681318681318
# 4. That is not the photo from Chelsea’s official account. Trump is in the photo in the photo on the official Chelsea account.     https://x.com/chelseafc/status/1944522524848763278?s=46 | Published on Mon Jul 14 22:35:43 +0000 2025; Helpfulness Score: 0.5196850393700787"""

# **Super Community Note:**