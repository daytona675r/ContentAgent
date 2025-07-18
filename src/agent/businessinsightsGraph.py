#Load environment variables from .env file
import json
import os
from typing import Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
import langchain_google_genai
import requests 
from src.agent.prompts import SUMMARIZE_BUSINESS_INPUT_PROMPT, SUMMARIZE_INSIGHTS_PROMPT
from tavily import TavilyClient

load_dotenv()

#State
class BusinessInsightsState(TypedDict):
    business_input: str
    summarized_business: Optional[str]
    insight_queries: Optional[List[str]]
    raw_insight_results: Optional[List[str]]  # list of answer strings
    summarized_insights: Optional[Dict[str, str]]  # query -> summary

initial_state: BusinessInsightsState = {
    "business_input": "",
    "summarized_business": None,
    "insight_queries": None,
    "raw_insight_results": None,
    "summarized_insights": None
}

# Initialize OpenAI client
ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
# Set Google API key for Gemini
langchain_google_genai.GoogleGenerativeAI.api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM Selection Helper ---
def get_llm(state: BusinessInsightsState, temperature: float = 0.8, top_p: float = 0.95):
    model_choice = state.get("model_choice", "gpt-4")
    # Map UI model names to API model names
    model_map = {
        "openai gpt-4": "gpt-4",
        "gemini 1.5 pro": "gemini-1.5-pro"
    }
    model_name = model_map.get(model_choice.lower(), "gpt-4")
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model_name=model_name, temperature=temperature, top_p=top_p)
    else:
        return langchain_google_genai.ChatGoogleGenerativeAI(model=model_name, temperature=temperature, top_p=top_p)

def search_tavily(query: str, api_key: str, site: str = None):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "include_answer": True,
        "max_results": 5,
        "search_depth": "advanced"
    }

    if site:
        payload["include_domains"] = [site]

    response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "")
    return ""


#Graph nodes

#store business info in vector db
def store_business_info(state: BusinessInsightsState) -> BusinessInsightsState:
    business_store = Chroma(
        collection_name="business_info",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    metadata = {
        "source": "user_input",
        "type": "business_info",
    }

    business_store.add_texts([state["business_input"]], metadatas=[metadata])

    return state  # Unchanged state

# 1. Summarize business input
def summarize_business(state: BusinessInsightsState) -> BusinessInsightsState:
    temperature = state.get("temperature", 0.8)
    top_p = state.get("top_p", 0.95)
    llm = get_llm(state, temperature=temperature, top_p=top_p)
    prompt=SUMMARIZE_BUSINESS_INPUT_PROMPT.format(
        business_input=state["business_input"][:12000] # Truncate to 12,000 characters
    )
    summary = llm.invoke(prompt)
    json_summary = json.loads(summary.content)
    return {**state, "summarized_business": json_summary}

# 2. Generate research queries
def generate_queries(state: BusinessInsightsState) -> BusinessInsightsState:
    summary = state["summarized_business"]
    industry = summary.get("Industry / Niche", "the business")
    product = summary.get("Product or Service Description", "the product")
    target = summary.get("Target Audience", "customers")

    queries= [
        f"What are common pain points for {target} in the {industry} industry?",
        f"What market trends are currently shaping the {industry} sector?",
        f"What competing solutions exist for {product}?",
        f"What kind of content engages {target} in the {industry} space?",
        f"What feedback or issues do users have with similar products to {product}?",
        f"What are misconceptions or hesitations {target} might have regarding {product}?",
        f"What unique angles could be used to position {product} in the market?"
    ]
    return {**state, "insight_queries": queries}

 # 3. Run Tavily search per query/platform
def get_social_insights(state: BusinessInsightsState) -> BusinessInsightsState:
    research_queries = state["insight_queries"]
    platforms = ["reddit.com", "quora.com", "twitter.com", "news.ycombinator.com"]

    all_answers = []

    for question in research_queries:
        for platform in platforms:
            result = search_tavily(question, os.getenv("TAVILY_API_KEY"), site=platform)
            if result:
                all_answers.append(f"From {platform} on '{question}':\n{result}\n")
    return {**state, "raw_insight_results": all_answers}

# 4. Summarize insights with LLM
def summarize_insights(state: BusinessInsightsState) -> BusinessInsightsState:
    temperature = 0.3
    top_p = state.get("top_p", 0.95)
    llm = get_llm(state, temperature=temperature, top_p=top_p)
    raw_insights = "\n".join(state["raw_insight_results"])[:12000]  # Truncate to 12,000 characters
    prompt=SUMMARIZE_INSIGHTS_PROMPT.format(
        raw_insights=raw_insights 
    )
    summary = llm.invoke(prompt)

    try:
        summary_dict = json.loads(summary.content)
    except Exception as e:
        summary_dict = {"error": f"Could not parse summary: {e}\nRaw output: {summary.content}"}
    return {**state, "summarized_insights": summary_dict}

#store summarized insights in vector db
def store_summarized_insights(state: BusinessInsightsState) -> BusinessInsightsState:
    insights_store = Chroma(
        collection_name="business_insights",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    for category, insight in state.get("summarized_insights", {}).items():
        # Flatten lists to a single string
        if isinstance(insight, list):
            insight = " ".join(str(i) for i in insight)
            # Flatten dicts to a single string
        elif isinstance(insight, dict):
            insight = " ".join(f"{k}: {v}" for k, v in insight.items())
        metadata = {
            "category": category,
            "type": "insight",
            "source": "auto_summary"
        }
        insights_store.add_texts([insight], metadatas=[metadata])

    return state  # Unchanged state

#Building the graph
workflow = StateGraph(BusinessInsightsState)
workflow.add_node("SummarizeBusiness", summarize_business)
workflow.add_node("StoreBusinessInfo", store_business_info)
workflow.add_node("GenerateQueries", generate_queries)
workflow.add_node("GetSocialInsights", get_social_insights)
workflow.add_node("SummarizeInsights", summarize_insights)
workflow.add_node("StoreSummarizedInsights", store_summarized_insights)

# Entry point
workflow.set_entry_point("StoreBusinessInfo")
workflow.add_edge("StoreBusinessInfo", "SummarizeBusiness")
workflow.add_edge("SummarizeBusiness", "GenerateQueries")
workflow.add_edge("GenerateQueries", "GetSocialInsights")
workflow.add_edge("GetSocialInsights", "SummarizeInsights")
workflow.add_edge("SummarizeInsights", "StoreSummarizedInsights")
workflow.add_edge("StoreSummarizedInsights", END)

insight_graph = workflow.compile()