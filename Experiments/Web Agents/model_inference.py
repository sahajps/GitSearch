from openai import OpenAI
from google import genai
from google.genai import types
from xai_sdk import Client as xaiClient
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters
import requests
import time
import subprocess

# OpenAI Client Setup
client_openai = OpenAI(api_key=open("../../Config/openai_key.txt", encoding="utf-8").read())

# Google GenAI Client Setup
client_google = genai.Client(api_key=open("../../Config/google_key.txt", encoding="utf-8").read())

# XAI Client Setup
client_xai = xaiClient(api_key=open("../../Config/xai_key.txt", encoding="utf-8").read())

# Perplexity API Setup
api_perplexity = open("../../Config/perplexity_key.txt", encoding="utf-8").read()
headers_perplexity = {
    "Authorization": f"Bearer {api_perplexity}",
    "Content-Type": "application/json"
}

###############################################################################
def generate_with_openai(prompt, model_name="gpt-5-nano-2025-08-07"):
    try:
        response = client_openai.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}],
            input=prompt
        )
    except:
        time.sleep(60)
        response = client_openai.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}],
            input=prompt
        )

    return response

################################################################################
def get_final_url(url):
    try:
        response = subprocess.run(
            ["curl", "-Ls", "-o", "/dev/null", "-w", "%{url_effective}", url],
            capture_output=True,
            text=True,
            timeout=150
        )
    except:
        return None
    return response.stdout

# For some URLs, this one was used instead of the above function
# But it was found to be slower and sometime stuck/not responding even with a higher timeout
# def get_final_url(url):
#     try:
#         # Use a HEAD request as it's more efficient; we only need the headers, not the content.
#         # timeout is added as a good practice to prevent the script from hanging.
#         response = requests.head(url, allow_redirects=True, timeout=5)
#         return response.url
#     except requests.exceptions.RequestException as e:
#         return None
    
# This function is directly taken from here: https://ai.google.dev/gemini-api/docs/google-search
def add_citations_gemini(response):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    # Keeping a single list of citations to append at the end, not in between the text
    citation_links = []
    for support in sorted_supports:
        # end_index = support.segment.end_index # Not used currently
        if support.grounding_chunk_indices:
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    # This is a bit changed
                    uri = get_final_url(chunks[i].web.uri)
                    # Even if get_final_url fails, we can still add the original reference link by the model (we don't want information loss)
                    if uri==None:
                        uri = chunks[i].web.uri
                    citation_links.append(uri)

    # Adding URLs at the end of the response text
    citation_string = " ".join(citation_links)
    text += f" {citation_string}"
    text = text.strip()

    return text

def generate_with_google(prompt, model_name="gemini-2.5-flash"):
    grounding_tool_google = types.Tool(google_search=types.GoogleSearch())
    config_google = types.GenerateContentConfig(tools=[grounding_tool_google])
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

################################################################################
def generate_with_xai(prompt, model_name="grok-4-0709"):
    try:
        chat = client_xai.chat.create(
            model=model_name,
            search_parameters=SearchParameters(mode="auto"),
        )
        chat.append(user(prompt))
        response = chat.sample()
    except:
        time.sleep(60)
        chat = client_xai.chat.create(
            model=model_name,
            search_parameters=SearchParameters(mode="auto"),
        )
        chat.append(user(prompt))
        response = chat.sample()
    
    return response

################################################################################
def generate_with_perplexity(prompt, model_name="sonar-deep-research"):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers_perplexity, timeout=150)
        assert response.status_code == 200, f"Request failed: {response.status_code}"
    except:
        time.sleep(60)
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers_perplexity, timeout=300)

    return response.json()