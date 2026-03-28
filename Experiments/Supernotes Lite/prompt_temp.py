# Refined by Sahaj
# Prompt template for experiments

from datetime import datetime, UTC

def timeToDate(createdAt):
  # fill that here

  # Convert to seconds
  seconds = createdAt / 1000.0

  # Convert to UTC datetime
  dt_utc = datetime.fromtimestamp(seconds, UTC)

  # Format like "Wed Apr 24 16:31:13 +0000 2024"
  date = dt_utc.strftime("%a %b %d %H:%M:%S +0000 %Y")

  return date

# For Supernote Lite.
def prompt_for_supernote_lite(tweet_text, date, comm_notes):
    comm_notes_prompt = "\n".join([
      f"{idx+1}. {note} | Published on {timeToDate(createAt)}; Helpfulness Score: {helpfulnessRatio}" for idx, (note, createAt, helpfulnessRatio) in enumerate(comm_notes)
    ])

    prompt = f"""You are an expert fact-checker. X (Twitter) has a crowd-sourced fact-checking program, called Community Notes. Here, users can write 'notes' on potentially misleading tweets. Each note needs to be rated helpful by a sufficient number of diversely-opinionated people (note-raters) for it to be shown publicly alongside the piece of content.

Helpful attributes in notes include:
- Cites high-quality sources
- Easy to understand
- Directly addresses the post's claim
- Provides important context
- Neutral or unbiased language

Unhelpful attributes in notes include:
- Sources not included or unreliable
- Sources do not support note
- Incorrect information
- Opinion or speculation
- Typos or unclear language
- Misses key points or irrelevant
- Argumentative or biased language
- Note not needed on this post
- Harassment or abuse

### Task
You will be given a potentially misleading tweet, its existing community notes, and the helpfulness scores (0-1) of those notes. Your task is to create a single helpful **Super Community Note** that balances the perspectives reflected in the helpfulness scores while summarizing the key points from the notes. The Supernote should provide clear, factual context that addresses the potentially misleading information in the tweet and be strong enough to replace all existing notes.
- The note must be within 280 characters. Treat each URL as 1 character, regardless of its actual length.
- The note must include one or more URLs to credible sources. Use only URLs explicitly provided in context. Do not invent or substitute other URLs.
- The note must be neutral, factual, and concise. When possible, cite sources across the political spectrum to strengthen neutrality, but prioritize reliability and relevance.
- Output only the Super Community Note text, with URLs included. Do not add explanations, formatting, or extra commentary.
- Do not include any information beyond what is explicitly provided in the context.

---
**Tweet (published on {date}):**
\"\"\"{tweet_text}\"\"\"

**Existing Community Note(s):**
\"\"\"{comm_notes_prompt}\"\"\"

**Super Community Note:**
"""
    return prompt