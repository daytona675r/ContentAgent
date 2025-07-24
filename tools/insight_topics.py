import json
import os
import langchain_google_genai
from langchain_openai import ChatOpenAI
from agent.prompts import TOPICS_FROM_INSIGHTS_PROMPT
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


insights_store = Chroma(
    collection_name="business_insights",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Initialize OpenAI client
ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
# Set Google API key for Gemini
langchain_google_genai.GoogleGenerativeAI.api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM Selection Helper ---
def get_llm(model: str, tone: str, temperature: float = 0.8, top_p: float = 0.95):
    model_choice = model
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
    

def get_market_insights():
    # Get all docs with type 'insight'
    docs = insights_store.get(where={"type": "insight"})
    # Extract all insights as a list
    if docs and docs['documents']:
        return docs['documents']  # list of strings
    return []

def get_market_insight_topics(model: str, tone: str, temperature: float = 0.8, top_p: float = 0.95):
    llm = get_llm(model=model, tone=tone, temperature=temperature, top_p=top_p)
    prompt=TOPICS_FROM_INSIGHTS_PROMPT.format(
        market_insights=get_market_insights(),
        tone=tone
    )
    result = llm.invoke(prompt)
    topics = json.loads(result.content)
    return topics