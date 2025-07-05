# tools/trending.py
import requests

def get_trending_topics(query="artificial intelligence", limit=5):
    API_URL = "https://api.search1api.com/news"

    data = {
        "query": query,
        "search_service": "google",
        "max_results": limit,
        "crawl_results": 2,
        "image": False,
        "include_sites": ["techcrunch.com"],
        "exclude_sites": ["wikipedia.org"],
        "language": "en",
        "time_range": "month"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        results = response.json()
        return [r["title"] for r in results.get("results", [])]
    except Exception as e:
        print("Failed to fetch trending topics:", e)
        return []
