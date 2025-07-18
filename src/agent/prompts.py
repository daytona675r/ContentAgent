# Prompt for LLM-based scoring of tweet variants (for score_and_select)
LLM_VARIANT_SCORER_PROMPT = '''
You are an expert social media content strategist. Given the following tweet variants, rate each one on a scale from 0 to 10 for the following criteria:
- Quality (writing, grammar, style)
- Relevance to the theme: {theme}
- Engagement potential (likelihood to get likes/retweets)
- Clarity
- Presence of a call-to-action (CTA)

For each variant, provide a JSON object with the following fields:
  {{
    "variant": <the tweet>,
    "quality": <0-10>,
    "relevance": <0-10>,
    "engagement": <0-10>,
    "clarity": <0-10>,
    "cta": <0-10>
  }}
Return a JSON list of all variants scored as above. Do not add any commentary.

Tweet Variants:
{variants}
'''
TWITTER_PROMPT_BASE = '''
You're an expert content strategist helping AI solopreneurs build their personal brand on Twitter/X.

Write a **viral-style tweet** about the topic:  
**"{theme}"**

Make it:
- **Hooky** in the first line (to grab attention)
- Clear, concise, and **relatable**
- With **a takeaway, insight, or small epiphany**
- Optionally formatted as a **list, insight thread, or observation**
- Avoid hashtags or emojis
- Fit within **280 characters**
- Written in a {personality} tone for founders and builders

Use these **top-performing examples** for inspiration:  
{examples}

Respond with only the tweet — no commentary.
You are allowed to use emojis sparingly if they fit the topic.
'''

LINKEDIN_PROMPT_BASE = '''
You're a startup advisor helping AI solopreneurs build their personal brand on LinkedIn.

Take the following short tweet and rewrite it as a short, high-performing **LinkedIn post**:
    
Tweet: **"{tweet}"**"

Make sure to:
- Start with a **strong hook line** to stop the scroll (question, insight, or bold statement)
- Keep paragraphs short (1–3 lines max)
- Include **personal insight** or **lesson learned**
- Offer value, a relatable struggle, or an unexpected insight
- Write in a {personality} tone — not too formal
- End with a light call-to-action (e.g. “Curious how others handle this?” or “Would love your take.”)

Use the following **top-performing tweet examples** as inspiration, but expand into a more LinkedIn-appropriate style:  
{examples}

Respond with only the post content — no hashtags, no commentary.
You are allowed to use emojis sparingly if they fit the topic.
'''

LLM_JUDGE_PROMPT = '''
You are an expert social media strategist.

Your task is to evaluate the following tweet for its quality, clarity, and potential to perform well on Twitter/X:

        ---
        Tweet: **"{tweet}"**"
        ---

        Score it on a scale from 0 to 1 (float). Then decide if it should be 'approved' or needs a 'retry'.
        It should be approved if the scale is above 0.75.

        Guidelines:
        - Engaging opening (hook)
        - Relevance to tech/AI/startups
        - Clarity and tone for the audience
        - Emotional or curiosity-triggering phrasing
        - Approaches virality (based on past viral patterns)

        Respond with JSON like:
        {{"score": 0.83, "status": "approved"}}
'''

THEME_REFINER_PROMPT = '''
You are a social media strategist. Given the following user input, extract or rewrite it as a clear, concise, and valuable tweet theme or idea suitable for AI/tech/solopreneur audiences.

User input: "{user_input}"

Respond with only the improved theme.
'''

SUMMARIZE_BUSINESS_INPUT_PROMPT = '''
    You are a startup analyst. Analyze the following business information and extract a structured summary:

    - Industry / Niche
    - Target Audience
    - Product or Service Description
    - Problem it Solves
    - Key Features or Keywords
    - Unique Selling Points (USP)

    Respond in JSON format.

    --- Business Info ---
    {business_input}
    '''

SUMMARIZE_INSIGHTS_PROMPT = '''
  You are an expert business analyst. Given the following raw insights from social media discussions and answers,
  summarize them into key categories.

  Return your answer as a JSON object where each key is a category (e.g., "Market Trends", "Customer Pain Points", 
  "Competitive Landscape", "Content Strategy Suggestions") and each value is a concise insight for that category.

  Respond ONLY with valid JSON. Do not include any commentary or explanation.

  --- Raw Insights ---
  {raw_insights}
  --- End Raw Insights ---
'''