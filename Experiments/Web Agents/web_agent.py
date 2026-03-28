import json
from prompt_temp import zero_shot_prompt_for_close_websearch_llms
from model_inference import generate_with_openai, generate_with_google, add_citations_gemini, generate_with_xai, generate_with_perplexity
from tqdm import tqdm
import traceback
import sys
import os

os.makedirs("../Outputs", exist_ok=True)
os.makedirs("Logs", exist_ok=True)

model_type = sys.argv[1] # "gpt-5-nano" or "gemini-2.5-flash"
run_time = sys.argv[2]  # "T1", "T2", ...

# Load tweet data
dic_tweets = json.load(open("../../Data/tweets.json", encoding="utf-8"))

test_help = json.load(open("../../Data/test_helpful.json", encoding="utf-8"))

tweetIds = test_help["tweetId"]

# Prompt building
prompts = []

for id in tweetIds:
    txt = dic_tweets[id]["text"]
    date = dic_tweets[id]["createdAt"]

    prompts.append( zero_shot_prompt_for_close_websearch_llms(txt, date) )

# Calling web-agents to write Community Notes for the above tweets...
try:
    CNs = json.load(open(f"../Outputs/web_search_{model_type}_{run_time}.json", encoding="utf-8"))
    Logs = json.load(open(f"Logs/logs_{model_type}_{run_time}.json", encoding="utf-8"))
except:
    CNs = {}
    Logs = {}

try:
    if model_type == "gpt-5-nano":
        for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                resp = generate_with_openai(pr)

                CNs[id] = resp.output[-1].content[0].text
                Logs[id] = {"prompt": pr, "response": str(resp)}

    elif model_type == "gemini-2.5-flash":
        for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                resp = generate_with_google(pr)

                # Try to add citations, skiping otherwise -- say if grounding not present etc.
                try:
                    CNs[id] = add_citations_gemini(resp)
                except:
                    CNs[id] = resp.text
                Logs[id] = {"prompt": pr, "response": str(resp)}

    elif model_type == "grok-4":
        for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                resp = generate_with_xai(pr)

                CNs[id] = ( resp.content + " Add-ons: " + " ".join(resp.citations) ).strip()
                Logs[id] = {"prompt": pr, "response": str(resp)}

    elif model_type == "sonar-deep-research":
        for id, pr in tqdm(zip(tweetIds, prompts), total=len(tweetIds)):
            if id not in CNs:
                resp = generate_with_perplexity(pr)

                CNs[id] = ( resp['choices'][0]['message']['content'] + " Add-ons: " + " ".join(resp['citations']) ).split("</think>")[-1].strip()
                Logs[id] = {"prompt": pr, "response": str(resp)}

    else:
        exit("This model is not supported currently.")
except Exception as e:
    print("Some error occurred. Saving results so far...")
    traceback.print_exc()

# Save results
json.dump(CNs, open(f"../Outputs/web_search_{model_type}_{run_time}.json", "w", encoding="utf-8"), indent=4)
json.dump(Logs, open(f"Logs/logs_{model_type}_{run_time}.json", "w", encoding="utf-8"), indent=4)



# Sample prompt:-
# You are an expert fact-checker with the ability to use web search tool, enabling you to verify information and write accurate notes to debunk misinformation. X (Twitter) has a crowd-sourced fact-checking program, called Community Notes. Here, users can write 'notes' on potentially misleading tweets. Each note needs to be rated helpful by a sufficient number of diversely-opinionated people (note-raters) for it to be shown publicly alongside the piece of content.

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
# Write a helpful **Community Note** that clarifies or contextualizes the potentially misleading information in the tweet by providing additional context.
# - The note must be within 280 characters. Treat each URL as 1 character, regardless of its actual length.
# - The note must include one or more URLs to credible sources.
# - The note must be neutral, factual, and concise. When possible, cite sources across the political spectrum to strengthen neutrality, but prioritize reliability and relevance.
# - Output only the Community Note text, with URLs included. Do not add explanations, formatting, or extra commentary.

# ---
# **Tweet (published on Wed Aug 20 19:08:20 +0000 2025):**
# """Von Der Liar was apparently asked to "leave the room" as President Trump-

# "Only wanted to talk to leaders" 

# Look at her face, Utterly humiliated. https://t.co/OpsxQNXtXu"""

# **Community Note:**