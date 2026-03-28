# Prompt template for experiments

from datetime import datetime, UTC

def timeToDate(createdAt):
  # Convert to seconds
  seconds = createdAt / 1000.0

  # Convert to UTC datetime
  dt_utc = datetime.fromtimestamp(seconds, UTC)

  # Format like "Wed Apr 24 16:31:13 +0000 2024"
  date = dt_utc.strftime("%a %b %d %H:%M:%S +0000 %Y")

  return date

# For gap and contradiction identification
def prompt_gap_identification(tweet_text, date, notes_text):
    prompt = f"""You are an expert fact-checker and detective analyzing potentially misleading tweets and community notes to find gaps and conflicts.

### Task
Analyze the given potentially misleading tweet and associated community notes (if any) to identify gaps in information, contradictions, or areas needing further investigation. Your goal is to produce a structured list of gaps that require targeted searches to verify facts and provide additional context. Categorize each gap into ONE of these types:

UNSUBSTANTIATED_CLAIM: Factual claims made without sources or evidence
Example: "Studies show X" but no studies are cited
If no notes exist: The tweet makes a factual claim that requires verification.

CONTRADICTION: Conflicting information
Example: Note 1 says "increases" but Note 2 says "decreases"
If no notes exist: The tweet contains internal contradictions or logical fallacies.

VAGUE_REFERENCE: Non-specific references that should be made concrete
Example: "some studies", "recent reports"

MISSING_CONTEXT: Statistics, numbers, or claims lacking necessary context
Example: "Crime increased 50%" without baseline, timeframe, or location

SOURCE_VERIFICATION: Sources mentioned but not provided, or sources need verification
Example: "According to Harvard study" but no link or citation

MISSING_COVERAGE: Important aspects of the tweet not addressed
Example: Tweet makes 3 claims but notes only address 1
If no notes exist: The entire tweet implies a narrative that lacks context or factual checks.

### Output Format
Return a JSON array of gaps. Each gap should have:

gap_type: One of the 6 types above (exact string match)
description: Clear, specific explanation of the gap (1-2 sentences)
priority: Integer 1-5, where:
5 = Critical (factual claims without sources, major contradictions)
4 = High (important context missing, vague references to studies)
3 = Medium (minor missing context, secondary claims unsourced)
2 = Low (stylistic improvements, minor details)
1 = Very low (nice-to-have information)
suggested_query: Specific, targeted search query to fill this gap (be precise)

### Important Guidelines
NO NOTES SCENARIO: If "EXISTING COMMUNITY NOTES" is empty or "None", treat every factual claim in the tweet as a potential UNSUBSTANTIATED_CLAIM or MISSING_CONTEXT gap needing a search query.
Be strategic: Prioritize gaps that most impact credibility and completeness.
Be specific: "Study mentioned but not identified" is better than "needs more info"
Be actionable: Each gap should have a clear, searchable query.

---
**Tweet (published on {date}):**
\"\"\"{tweet_text}\"\"\"

**Existing Community Note(s):**
\"\"\"{notes_text}\"\"\"

### Your Response
Analyze the input above and return ONLY the JSON array. No preamble, no explanation, just the JSON.
"""
    return prompt

# For targeted search generation
def prompt_targeted_search_article(tweet_text, date, notes_text, gap_context):
    prompt = f"""You are an expert fact-checker with the ability to use web search tool, enabling you to verify information and write accurate fact-checking articles to debunk misinformation.

Helpful attributes in fact-checking include:
- Cites high-quality sources
- Easy to understand
- Directly addresses the post's claim
- Provides important context
- Neutral or unbiased language

Unhelpful attributes in fact-checking include:
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
Analyze the given potentially misleading tweet and community notes (if any), and use identified information gaps for targeted web searches to retrieve relevant facts from credible sources and synthesize the provided inputs into a short, authoritative fact-checking article. The article should directly address the misleading claims in the tweet, filling the identified gaps with verified information from reliable sources.

---
**Tweet (published on {date}):**
\"\"\"{tweet_text}\"\"\"

**Existing Community Note(s):**
\"\"\"{notes_text}\"\"\"

**Identified Gaps and Suggested Queries for Effective Fact-checking:**
\"\"\"{gap_context}\"\"\"

**Short Fact-checking Article:**
"""
    return prompt

# Final community notes synthesis
def prompt_synthesize_final_note(tweet_text, date, search_article):
    prompt = f"""You are an expert fact-checker and skilled writer dedicated to producing clear, accurate, and unbiased community notes that debunk misinformation.

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
You will be given a potentially misleading tweet and a fact-checking article. Your task is to create a helpful Community Note that balances perspectives reflected in helpfulness and focuses on maximizing the completeness of the note. The note should provide clear, factual context that addresses the potentially misleading information in the tweet and include full-text URLs to support factual information.
- The note must be within 280 characters. Treat each URL as 1 character, regardless of its actual length.
- The note must include one or more URLs to credible sources.
- The note must be neutral, factual, and concise. When possible, cite sources across the political spectrum to strengthen neutrality, but prioritize reliability and relevance.
- Output only the Community Note text, with URLs included. Do not add explanations, formatting, or extra commentary.
- Do not include any information beyond what is explicitly provided in the context.

---
**Tweet (published on {date}):**
\"\"\"{tweet_text}\"\"\"

**Fact-checking Article:**
\"\"\"{search_article}\"\"\"

**Community Note:**
"""
    return prompt