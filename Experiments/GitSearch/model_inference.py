from openai import OpenAI
from google import genai
from google.genai import types
import time
import subprocess

# OpenAI Client Setup
client_openai = OpenAI(api_key=open("../../Config/openai_key.txt", encoding="utf-8").read())

# Google GenAI Client Setup
client_google = genai.Client(api_key=open("../../Config/google_key.txt", encoding="utf-8").read())

###############################################################################
def generate_with_openai(prompt, model_name="gpt-5-nano-2025-08-07", use_web_search=False):
    if use_web_search:
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
    else:
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
    
# This function is directly taken from here: https://ai.google.dev/gemini-api/docs/google-search
def add_citations(response):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    # This is a bit changed
                    uri = get_final_url(chunks[i].web.uri)
                    # Even if get_final_url fails, we can still add the original reference link by the model (we don't want information loss)
                    if uri==None:
                        uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text

def generate_with_google(prompt, model_name="gemini-2.5-flash", use_web_search=False):
    if use_web_search:
        grounding_tool_google = types.Tool(google_search=types.GoogleSearch())
        config_google = types.GenerateContentConfig(tools=[grounding_tool_google])
    else:
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