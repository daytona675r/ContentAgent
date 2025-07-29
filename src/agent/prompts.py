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

TOPICS_FROM_INSIGHTS_PROMPT = '''
  You are a content strategist helping a startup create social media content that resonates with its target audience.


  Here are insights from the market and customer conversations:
  {market_insights}

  Please generate 5 content topic ideas that:
  - Address key pain points or challenges discussed in the market
  - Reflect current trends or opportunities
  - Are relevant to the target audience
  - Align with the product or positioning of the startup
  - Compare our product with others in the competitive landscape
  - Simply use the Content Strategy Suggestions from the market insights

  Keep the tone {tone}, and write each topic as a short, punchy title or concept idea.
  Return the topics as a json list.
'''

PLATFORM_CONTENT_PROMPT = '''
    You are an expert multi-platform social media content creator.

    ### Context:
    - **Market Insights**: {insights}
    - **Selected Topic**: {topic}

    ### Task:
    Generate **FOUR distinct social media posts** optimized for each platform:
    1. **Twitter (X)**
      - Short, punchy (max 280 characters)
      - Focus on virality: wit, bold statements, quick hooks
      - Use 1-3 relevant hashtags
    2. **LinkedIn**
      - Professional and thought-leadership tone (between 1300 and 2000 characters)
      - Emphasize credibility, insight, and industry relevance
    3. **Facebook**
      - Conversational, community-oriented (between 40 and 80 characters)
      - Include a relatable hook or story
      - Encourage comments/shares
    4. **Instagram**
      - Visual or meme-like appeal
      - Short caption with emotional/aspirational vibe
      - Include 2-3 trendy hashtags

    ### Requirements:
    - Capture attention quickly.
    - Posts must feel **different per platform** (not copies).
    - Adapt tone and content to **audience expectations** of each platform.
    - Align with the company’s product and audience pain points.
    - Use **insights** to highlight relevance and value.
    - Return **strict JSON** with keys `twitter`, `linkedin`, `facebook`, `instagram`. Put the content in the post field and the hashtags in the hashtags field.

    ### JSON Output Format (must be valid):
    {{
      "twitter": {{
        "post": "string",
        "hashtags": "#tag1", "#tag2"
      }},
      "linkedin": {{
        "post": "string",
        "hashtags": "#tag1", "#tag2"
      }},
      "facebook": {{
        "post": "string",
        "hashtags": "#tag1", "#tag2"
      }},
      "instagram": {{
        "post": "string",
        "hashtags": "#tag1", "#tag2"
      }}
    }}

    Only return the JSON, no explanations.
"""

'''

PERSONA_SCORE_PROMPT = '''
    You are evaluating social media content drafts. Four personas will provide their perspective:
    1. Social Media Strategist – values clarity, engagement, virality potential, platform tone, and brand alignment.
    2. Industry Expert – values technical accuracy, relevance, and thought leadership.
    3. Target Customer – values pain point resonance, clarity of solution, and helpfulness.
    4. Investor – values market potential, professionalism, and scalability.

    For each persona, provide:
    - Score (0-10)
    - 2-3 bullet points of feedback

    Platform: {platform}
    Content Draft:
    {content}

    ### JSON Output Format (must be valid):
    {{
      "social_media_strategist": {{"score": 0-10, "feedback": "..."}},
      "industry_expert": {{"score": 0-10, "feedback": "..."}},
      "target_customer": {{"score": 0-10, "feedback": "..."}},
      "investor": {{"score": 0-10, "feedback": "..."}}
    }}

'''

REFINEMENT_PROMPT = """
  You are an expert social media copywriter skilled in adapting content for multiple platforms.

  ## Goal
  Refine the following post to address the feedback provided. Keep the platform-specific style intact while improving weak points.

  ### Inputs
  - **Platform:** {platform}
  - **Topic:** {topic}
  - **Current Draft:**
  {draft}

  - **Feedback (from multiple personas):**
  {feedback}

  ### Instructions
  - Fix weaknesses mentioned in the feedback.
  - Do not lose the strengths of the original draft.
  - Ensure platform style is respected:
    - Twitter: Short, snappy, viral.
    - LinkedIn: Professional, thought-leadership.
    - Facebook: Community-oriented, conversational.
    - Instagram: Visual, emotional, trendy.

  ### Output
  - Return **strict JSON**. Put the refined post in the post field and the hashtags if available in the hashtags field.

    ### JSON Output Format (must be valid):
    {{
      "post": "string",
      "hashtags": "#tag1", "#tag2"
    }}

    Only return the JSON, no explanations.
"""