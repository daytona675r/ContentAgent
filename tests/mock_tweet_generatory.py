import json
import random
from datetime import datetime, timedelta

# Mock themes and styles
themes = ["Startup Building", "Solopreneur Tactics", "AI Productivity", "Prompt Engineering", "No-Code AI"]
tones = ["casual", "professional", "insightful", "motivational"]
styles = ["listicle", "thread", "one-liner", "case study"]

# Tweet text generator (basic placeholders)
def generate_tweet_text(theme):
    starters = [
        f"Why most {theme.lower()} ideas fail (and what to do instead)",
        f"3 lessons I learned from building an AI-first tool",
        f"The solopreneur's roadmap for launching with AI",
        f"A quick breakdown of {theme.lower()} success",
        f"AI isn’t enough: {theme.lower()} needs this too"
    ]
    return random.choice(starters)

# Generate one mock tweet object
def generate_mock_tweet(i, date_offset):
    theme = random.choice(themes)
    text = generate_tweet_text(theme)
    likes = random.randint(5, 100)
    retweets = random.randint(0, 25)
    replies = random.randint(0, 10)
    impressions = likes * 30 + random.randint(-100, 200)

    date = (datetime.today() - timedelta(days=date_offset)).strftime("%Y-%m-%d")
    time_posted = f"{random.randint(7, 22)}:{random.choice(['00', '15', '30', '45'])}"

    return {
        "tweet_id": f"{i:03}",
        "text": text,
        "theme": theme,
        "date": date,
        "engagement": {
            "likes": likes,
            "retweets": retweets,
            "replies": replies,
            "impressions": impressions
        },
        "tone": random.choice(tones),
        "style": random.choice(styles),
        "time_posted": time_posted
    }

# Generate 100 mock tweets over the last 5 months
mock_data = [generate_mock_tweet(i, random.randint(0, 150)) for i in range(1, 101)]

# Save to file
with open("mock_tweet_history.json", "w") as f:
    json.dump(mock_data, f, indent=2)

print("✅ mock_tweet_history.json created with 100 entries.")
