import re
import pandas as pd
from openai import OpenAI
import time
import json
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import statistics as stats
import numpy as np
from urlextract import URLExtract
from urllib.parse import urlparse
from transformers import pipeline, set_seed
url_extractor = URLExtract()
set_seed(42)

# OpenAI Client Setup
client_openai = OpenAI(api_key=open("../Config/openai_key.txt", encoding="utf-8").read())

###############################################################################
def generate_with_openai(prompt, model_name="gpt-5.2-2025-12-11"):
    try:
        response = client_openai.responses.create(
            model=model_name,
            input=prompt
        )
        return json.loads(response.output[-1].content[0].text)
    except:
        time.sleep(60)
        response = client_openai.responses.create(
            model=model_name,
            input=prompt
        )
        return json.loads(response.output[-1].content[0].text)

############# Notes Text Stats & Processing #############
def note_length_and_urls(notes):
    lengths = []
    num_urls = []
    for n in notes:
        urls = url_extractor.find_urls(n)
        num_urls.append( len(urls) )

        txt = n
        for u in urls:
            txt = txt.replace(u, 'U')
        lengths.append( len(txt) )

    return lengths, num_urls

def remove_links_from_notes(notes):
    notes_without_urls = []
    for n in notes:
        urls = url_extractor.find_urls(n)
        txt = n
        for u in urls:
            txt = txt.replace(u, ' ')
        txt = re.sub(r'\s+', ' ', txt).strip()
        notes_without_urls.append(txt)
        
    return notes_without_urls

def remove_NA_from_notes(references, predictions):
    new_ref, new_pred = [], []

    for r, p in zip(references, predictions):
        if (r!="NA") and (p!="NA"):
            new_ref.append(r)
            new_pred.append(p)

    return new_ref, new_pred

############# ROUGE-L #############
def compute_rouge_l(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    fmeasures = []
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref.lower(), pred.lower())["rougeL"]
        fmeasures.append(scores.fmeasure)

    return fmeasures

############# BERTScore #############
def compute_bert_score(references, predictions):
    P, R, F1 = bertscore(predictions, references, lang="en")
    
    return F1.tolist()

############# URL Recall #############
def compute_url_recall(references, predictions):
    recall_list = []
    for r, p in zip(references, predictions):
        r_urls = set(url_extractor.find_urls(r))
        p_urls = set(url_extractor.find_urls(p))

        if not r_urls:
            recall_list.append(1) # avoid division by zero
            continue
        true_pos = len(r_urls & p_urls)
        recall_list.append( true_pos/len(r_urls) )

    return recall_list

############# Domain Bias #############
def compute_domain_bias_scores_lowcred(notes):
    df = pd.read_csv("../Data/domain_bias_scores.csv")
    df = df.dropna(subset=["domain"])
    df = df.drop_duplicates(subset=["domain"], keep="first")
    df.index = df.domain

    def domain_finder(note):
        urls = url_extractor.find_urls(note)
        domains = []
        for u in urls:
            url_dom = urlparse(u).netloc
            tmp_dom = []
            for d in df.domain:
                if d in url_dom:
                    tmp_dom.append(d)
            if len(tmp_dom) > 0:
                domains.append( max(tmp_dom, key=len) )
                
        return domains

    ideology_score = []
    is_lowcred = []
    for n in notes:
        domains = domain_finder(n)
        for d in domains:
            tmp = df.loc[d]
            if not np.isnan(tmp['ideology_scaled']):
                ideology_score.append( float(tmp['ideology_scaled']) )
            if not np.isnan(tmp['reconciled_lowcred']):
                is_lowcred.append( int(tmp['reconciled_lowcred']) )

    return ideology_score, is_lowcred

############# Language Bias #############
def compute_bias_in_langauge(notes):
    pipe = pipeline("text-classification", 
                model="mediabiasgroup/magpie-babe-ft-xlm",
                tokenizer="mediabiasgroup/magpie-babe-ft-xlm",
                truncation=True,   # cut off extra tokens
                max_length=512
                )

    pred = pipe(notes)
    bias_scores = []
    for p in pred:
        if p['label']=='biased':
            bias_scores.append(p['score'])
        else:
            bias_scores.append(1-p['score'])

    return bias_scores

############# LLM-as-a-Judge #############
def compute_llm_judge_scores(tweet_text, date, human_note, ai_note):
    prompt = f"""You are an expert evaluator of community notes. Your task is to score an AI-generated note by comparing it to a human-written helpful note and the original tweet.

### Evaluation Criteria
Functional Errors (1–5): Evaluate whether the AI note has technical or usability issues that reduce its quality, including truncated or incomplete text, broken or incomplete URLs, formatting or punctuation problems, excessive length, or the presence of reasoning or meta-level commentary. A score of 5 means no functional issues at all, while a score of 1 means severe errors that significantly impair usability.

Claim Alignment (1–5): Evaluate how accurately the AI note identifies and addresses the primary claim or claims made in the tweet. The note should directly engage with what is actually being asserted or implied, focus on the aspects that require verification, and avoid shifting attention to related but different issues. Proper claim alignment also requires that the facts presented are relevant to the identified claim and help resolve it, rather than merely being topically similar. A score of 5 means the note precisely targets the core claim and supports it with relevant facts, while a score of 1 means the note misunderstands, ignores, or substitutes the main claim with an unrelated or tangential one.

Fact Alignment (1–5): Judge whether the factual statements in the AI note are consistent with the human-written note, focusing on whether they describe the same underlying facts, entities, events, timelines, and claims. Different sources are acceptable if they substantively support the same facts, but the alignment must also be verifiable through the cited URLs, including that the source content actually supports the specific statements being made. Topical similarity alone is not sufficient. A score of 5 means all factual claims are fully consistent with the human note and correctly supported by the sources, while a score of 1 means the note contains contradictions, incorrect references, unsupported claims, or factual errors.

Completeness (1–5): Evaluate whether the AI note fully covers the key facts and essential context needed to address the tweet’s main claim, as reflected in the human-written note. The note should include all information necessary to understand and verify the claim, without major omissions or reliance on vague or indirect references. Completeness also requires that included content is relevant to the core claim and factually aligned, rather than adding extraneous details. A score of 5 means the note is comprehensive, well-scoped, and factually aligned with the human note, while a score of 1 means critical facts or context related to the claim are missing, misaligned, or insufficient to properly evaluate the claim.

Helpfulness (1–5): Evaluate the overall usefulness of the AI note in helping a broad audience understand the post, following community notes standards. A helpful note clearly and directly addresses the post’s claim, uses neutral and unbiased language, and provides important context supported by high-quality, reliable sources that substantiate the stated facts. The note should be easy to understand and focused on information that is actually needed for this post. A score of 5 means the note is clear, well-sourced, relevant, and informative. A score of 1 means the note is unhelpful due to missing or unreliable sources, unsupported or incorrect information, unclear or poorly written text, irrelevant or missing key points, biased or argumentative language, or because the note is unnecessary or inappropriate for the post.

### Output Format
Return ONLY the following JSON structure:
{{
  "functional_errors": s1,
  "claim_alignment": s2,
  "fact_alignment": s3,
  "completeness": s4,
  "helpfulness": s5,
  "overall_comment": "<1–2 lines summarizing overall quality and suitability>"
}}

---
**Original Tweet (published on {date}):**
\"\"\"{tweet_text}\"\"\"

**Human-Written Helpful Note (Reference):**
\"\"\"{human_note}\"\"\"

**AI-Generated Note (To Evaluate):**
\"\"\"{ai_note}\"\"\"

### Your Response
Analyze the input above and return ONLY the JSON object. No preamble, no explanation, just the JSON.
"""
    
    return generate_with_openai(prompt)

############# Return Auto Scores #############
def return_auto_scores(references, predictions):
    references = [str(r) for r in references]
    predictions = [str(p) for p in predictions]
    scores = {}

    references, predictions = remove_NA_from_notes(references, predictions)

    scores['num_of_notes'] = len(predictions)

    scores['note_length'], scores['num_urls'] = note_length_and_urls(predictions)
    scores['url_recall'] = compute_url_recall(references, predictions)
    scores['domain_bias'], scores['is_lowcred'] = compute_domain_bias_scores_lowcred(predictions)

    predictions = remove_links_from_notes(predictions)
    references = remove_links_from_notes(references)

    scores['language_bias'] = compute_bias_in_langauge(predictions)
    scores['rouge_l'] = compute_rouge_l(references, predictions)
    scores['bert_score'] = compute_bert_score(references, predictions)

    return scores