import pandas as pd
import json
import os

os.makedirs("Outputs", exist_ok=True)

df_cn = pd.read_csv('../Data/notes.tsv', sep='\t', low_memory=False)

with open("../Data/tweets.json", "r") as f:
    dic_tw = json.load(f)

with open("../Data/test_helpful.json", "r") as f:
    test_help = json.load(f)

tweetIds = test_help["tweetId"]

CNs = {}
for id in tweetIds:
    tmp_notes = df_cn[df_cn['tweetId'] == int(id)]
    tmp_notes = tmp_notes[tmp_notes["helpfulnessStatus"] == "CURRENTLY_RATED_HELPFUL"]
    if len(tmp_notes) > 0:
        tmp_notes = tmp_notes.loc[tmp_notes["createdAtMillis"].idxmax()] # taking the most recent helpful note
        CNs[id] = tmp_notes["summary"]
    else:
        CNs[id] = None

with open("Outputs/human.json", "w") as f:
    json.dump(CNs, f, indent=4)