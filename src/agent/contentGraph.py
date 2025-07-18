from __future__ import annotations
from dataclasses import dataclass
import operator
from typing import Annotated, Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Add this import
from langchain_community.vectorstores import Chroma  # Import Chroma
import langchain_google_genai  # Import Google Generative AI

from langchain.prompts import PromptTemplate  # Import PromptTemplate
from src.agent.prompts import TWITTER_PROMPT_BASE, LINKEDIN_PROMPT_BASE, LLM_JUDGE_PROMPT, THEME_REFINER_PROMPT, LLM_VARIANT_SCORER_PROMPT  # Import prompts
from langchain_core.documents import Document  # Import Document
import tiktoken

#Load environment variables from .env file
load_dotenv()


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str





# Initialize OpenAI client
ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Set Google API key for Gemini
langchain_google_genai.GoogleGenerativeAI.api_key = os.getenv("GOOGLE_API_KEY")



# --- Vector DB Setup ---
tweet_store = Chroma(
    collection_name="tweet_history",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# --- Define LangGraph State ---

class ContentState(TypedDict):
    theme: str
    variant_ideas: List[str]
    top_performers: List[str]
    selected_idea: str
    linkedin_variant: str
    user_feedback: str
    score: float
    status: str  # "retry" or "approved"
    token_count: int  # Total tokens used in the graph run
    price_usd: float  # Estimated price in USD for the run

initial_state: ContentState = {
    "theme": "",
    "variant_ideas": [],
    "top_performers": [],
    "selected_idea": "",
    "linkedin_variant": "",
    "user_feedback": "",
    "score": 0.0,
    "status": "",
    "token_count": 0,
    "price_usd": 0.0
}

def store_top_performer(tweet: str, metadata: dict = None):
    print("💾 Storing tweet as Top Performer in Chroma DB...")
    if metadata is None:
        metadata = {"source": "agent", "score": 0.90}
    doc = Document(page_content=tweet, metadata=metadata)
    tweet_store.add_documents([doc])



# --- Shared Prompt Logic ---
def build_Twitter_prompt(state: ContentState) -> str:
    personality = state.get("personality", "smart-casual")
    return TWITTER_PROMPT_BASE.format(
        theme=state["theme"],
        examples="\n".join(state["top_performers"]),
        personality=personality
    )

def build_LinkedIn_prompt(state: ContentState) -> str:
    personality = state.get("personality", "smart, personal, and reflective")
    return LINKEDIN_PROMPT_BASE.format(
         tweet=state["selected_idea"],
        examples="\n".join(state["top_performers"]),
        personality=personality
    )

def build_LLM_Judge_prompt(state: ContentState) -> str:
    return LLM_JUDGE_PROMPT.format(
        tweet=state["selected_idea"]
    )

def build_theme_refiner_prompt(user_input: str) -> str:
    return THEME_REFINER_PROMPT.format(user_input=user_input)

# --- Nodes ---
def idea_selector(state: ContentState):
    print("🧠 IdeaSelector: Receiving theme input...")
    user_input = state.get("theme", "")
    if not user_input:
        state["theme"] = "Building AI-first solopreneur products"
        return state
    # LLM prompt for theme refinement
    prompt = build_theme_refiner_prompt(user_input)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    result = llm.invoke(prompt)
    refined_theme = result.content.strip()
    print(f"Refined theme: {refined_theme}")
    state["theme"] = refined_theme
    return state

def memory_retriever(state: ContentState):
    print("📚 Retrieving top-performing tweets from memory...")
    query = state["theme"]
     # Search the vectorstore
    results = tweet_store.similarity_search(query, k=10)  # Get more for filtering
    
    # Filter by likes > 30
    top_performers = []
    for r in results:
        likes = r.metadata.get("likes", 0)
        if likes > 30:
            top_performers.append(r.page_content)

    # Just return the top 3 if available
    state["top_performers"] = top_performers[:3]

    print(f"✅ Retrieved {len(top_performers[:3])} top tweets for inspiration.")
    return state

def count_tokens_gpt4(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_variants(state: ContentState):
    print("🔠 Generating 3 tweet variants...")
    state["variant_ideas"] = []
    temperature = state.get("temperature", 0.8)
    top_p = state.get("top_p", 0.95)
    llm = get_llm(state, temperature=temperature, top_p=top_p)
    total_tokens = 0
    for _ in range(3):
        prompt = build_Twitter_prompt(state)
        output = llm.invoke(prompt)
        content = output.content.strip()
        state["variant_ideas"].append(content)
        # Try to get token usage from output (if available)
        if hasattr(output, 'usage') and output.usage and 'total_tokens' in output.usage:
            total_tokens += output.usage['total_tokens']
        elif hasattr(output, 'token_count'):
            total_tokens += output.token_count
        else:
            # Estimate tokens for prompt + response
            total_tokens += count_tokens_gpt4(prompt) + count_tokens_gpt4(content)
    # Store tokens used so far
    state["token_count"] = state.get("token_count", 0) + total_tokens
    return state

def score_and_select(state: ContentState):
    print("🔎 Scoring and selecting best idea (LLM-based)...")
    if not state["variant_ideas"]:
        state["selected_idea"] = ""
        return state
   
    variants_str = "\n".join(f"{i+1}. {v}" for i, v in enumerate(state["variant_ideas"]))
    scoring_prompt = LLM_VARIANT_SCORER_PROMPT.format(theme=state["theme"], variants=variants_str)

    llm = get_llm(state, temperature=0)
    output = llm.invoke(scoring_prompt)

    try:
        scores = json.loads(output.content)
    except Exception as e:
        print("[ERROR] Failed to parse LLM scoring output:", e)
        print("Raw output was:\n", output.content)
        # Fallback: pick the longest
        selected = max(state["variant_ideas"], key=len)
        state["selected_idea"] = selected
        return state

    # Compute a composite score for each variant
    def composite_score(s):
        # Weighted sum: prioritize engagement, relevance, and quality
        return (
            2 * s.get("engagement", 0)
            + 2 * s.get("relevance", 0)
            + 1.5 * s.get("quality", 0)
            + 1.5 * s.get("clarity", 0)
            + 1 * s.get("cta", 0)
        )

    best = max(scores, key=composite_score)
    print(f"[score_and_select] Best variant score: engagement={best.get('engagement')}, relevance={best.get('relevance')}, quality={best.get('quality')}, clarity={best.get('clarity')}, cta={best.get('cta')}")
    state["selected_idea"] = best["variant"]
    return state

def generate_linkedin_variant(state: ContentState):
    print("🔗 Generating LinkedIn version of the tweet...")
    temperature = state.get("temperature", 0.7)
    top_p = state.get("top_p", 0.95)
    llm = get_llm(state, temperature=temperature, top_p=top_p)
    prompt = build_LinkedIn_prompt(state)
    output = llm.invoke(prompt)
    content = output.content.strip()
    state["linkedin_variant"] = content
    # Try to get token usage from output (if available)
    tokens = 0
    if hasattr(output, 'usage') and output.usage and 'total_tokens' in output.usage:
        tokens = output.usage['total_tokens']
    elif hasattr(output, 'token_count'):
        tokens = output.token_count
    else:
        tokens = count_tokens_gpt4(prompt) + count_tokens_gpt4(content)
    state["token_count"] = state.get("token_count", 0) + tokens
    # Calculate price (OpenAI GPT-4, as of July 2025, $0.03/1K prompt, $0.06/1K completion, use $0.06/1K as upper bound)
    state["price_usd"] = round((state["token_count"] / 1000) * 0.06, 4)
    return state

def score_and_feedback(state: ContentState, judge_model="gpt-3.5-turbo"):
    judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    print("🧑‍⚖️ LLM Judging the selected tweet...")

    prompt = build_LLM_Judge_prompt(state)

    result = judge_llm.invoke(prompt)

    print("[DEBUG] LLM Judge raw output:", result.content)  # <-- Debug print

    try:
        result_json = json.loads(result.content)
        state["score"] = result_json["score"]
        state["status"] = result_json["status"]
    except Exception as e:
        print("Judge parsing failed:", e)
        state["score"] = 0.5
        state["status"] = "retry"

    return state

def check_score(state: ContentState):
    return "end" if state["status"] == "approved" else "retry"

# --- LLM Selection Helper ---
def get_llm(state: ContentState, temperature: float = 0.8, top_p: float = 0.95):
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

# --- Main LangGraph ---
workflow = StateGraph(ContentState)
workflow.add_node("IdeaSelector", idea_selector)
workflow.add_node("MemoryRetriever", memory_retriever)
workflow.add_node("GenerateVariants", generate_variants)
workflow.add_node("ScoreAndSelect", score_and_select)
workflow.add_node("ScoreAndFeedback", score_and_feedback)

workflow.set_entry_point("IdeaSelector")
workflow.add_edge("IdeaSelector", "MemoryRetriever")
workflow.add_edge("MemoryRetriever", "GenerateVariants")
workflow.add_edge("GenerateVariants", "ScoreAndSelect")
workflow.add_edge("ScoreAndSelect", "ScoreAndFeedback")
workflow.add_node("GenerateLinkedInVariant", generate_linkedin_variant)

workflow.add_conditional_edges(
    "ScoreAndFeedback",
    check_score,
    {
        "retry": "GenerateVariants",
        "end": "GenerateLinkedInVariant"
    }
)

workflow.add_edge("GenerateLinkedInVariant", END)

graph = workflow.compile()

#final_state = graph.invoke({**initial_state,"theme": "Launching profitable AI solopreneur tools"})
#print("\n✅ Final Result:")
#print(final_state)  # Contains theme, all variants, selected, and feedback
