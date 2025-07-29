import os
import langchain_google_genai
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from dataclasses import dataclass, field
import json

from agent.prompts import PERSONA_SCORE_PROMPT, PLATFORM_CONTENT_PROMPT, REFINEMENT_PROMPT

MAX_REFINEMENTS = 1
THRESHOLD_SCORE = 8

# ---------------------------
# STATE DEFINITION
# ---------------------------
@dataclass
class ContentState:
    # Inputs
    topic: str
    context: str = ""
    summarized_insights: str = ""
    business_info: str = ""
    temperature: float = 0.8
    top_p: float = 0.95
    model: str = "gpt-4"  # default model choice
    current_node:str= ""
    
    # Generated drafts
    generated_content: Dict[str, str] = field(default_factory=dict)  # {platform: content}
    
    # Persona scoring
    persona_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Example: {"twitter": {"strategist": {"score": 7, "feedback": "..."} }}
    
    aggregated_scores: Dict[str, Any] = field(default_factory=dict)  # aggregated metrics per platform
    
    # Refinement tracking
    refinement_history: Dict[str, List[str]] = field(default_factory=dict)  # iterations per platform
    
    refinement_loops: dict = field(default_factory=lambda: {
        "twitter": 0, "facebook": 0, "linkedin": 0, "instagram": 0
    })

    low_score_platforms: list = field(default_factory=list)

    # Final chosen content
    final_content: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert state to dict for storage/UI rendering."""
        return {
            "topic": self.topic,
            "context": self.context,
            "summarized_insights": self.summarized_insights,
            "business_info": self.business_info,
            "generated_content": self.generated_content,
            "persona_scores": self.persona_scores,
            "aggregated_scores": self.aggregated_scores,
            "refinement_history": self.refinement_history,
            "final_content": self.final_content,
        }
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

#LLM Helpers

# Initialize OpenAI client
ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
# Set Google API key for Gemini
langchain_google_genai.GoogleGenerativeAI.api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM Selection Helper ---
def get_llm(state: ContentState, temperature: float = 0.8, top_p: float = 0.95):
    model_choice = state.model
    # Map UI model names to API model names
    model_map = {
        "openai gpt-4.5": "gpt-4.5",
        "gemini 1.5 pro": "gemini-1.5-pro"
    }
    model_name = model_map.get(model_choice.lower(), "gpt-4")
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model_name=model_name, temperature=temperature, top_p=top_p)
    else:
        return langchain_google_genai.ChatGoogleGenerativeAI(model=model_name, temperature=temperature, top_p=top_p)

def get_content_drafts(state: ContentState):
    llm = get_llm(state, temperature=state.temperature, top_p=state.top_p)

    prompt=PLATFORM_CONTENT_PROMPT.format(
        topic=state.topic,
        insights=state.summarized_insights,
    )
    return llm.invoke(prompt)

def judge_content(state: ContentState, platform: str, content: str):
    #llm = get_llm(state, temperature=state.temperature, top_p=state.top_p)
    llm=ChatOpenAI(model_name="gpt-4", temperature=state.temperature, top_p=state.top_p)
    prompt = PERSONA_SCORE_PROMPT.format(
        content=content,
        platform=platform
    )
    return llm.invoke(prompt)

def refine_content(state: ContentState, platform: str, combined_feedback: str):
    llm = get_llm(state, temperature=state.temperature, top_p=state.top_p)

    prompt = REFINEMENT_PROMPT.format(
        platform=platform,
        topic=state.topic,
        draft=state.generated_content.get(platform, ""),
        feedback=combined_feedback,
    )
    return llm.invoke(prompt)

# ---------------------------
# NODES
# ---------------------------

# 1. Generate content drafts per platform
def generate_content_node(state: ContentState):
    state.current_node = "generate_content_node"
    json_content = json.loads(get_content_drafts(state).content)
    state.generated_content = {
        "twitter": json_content.get("twitter", ""),
        "facebook": json_content.get("facebook", ""),
        "linkedin": json_content.get("linkedin", ""),
        "instagram": json_content.get("instagram", "")
    }
    # Initialize refinement history
    for platform in state.generated_content:
        state.refinement_history[platform] = [state.generated_content[platform]["post"]]
    return state


# 2. Score drafts via persona judges
def score_personas_node(state: ContentState):
    state.current_node = "score_personas_node"
    state.persona_scores = {}
    updated_scores = {}

    for platform, content in state.generated_content.items():
        state.persona_scores[platform] = {}
        # Call LLM to score + feedback
        score_feedback = judge_content(state, platform, content["post"])
        json_feedback=json.loads(score_feedback.content)
        updated_scores[platform] = json_feedback
        state.persona_scores = updated_scores
    return state


# 3. Aggregate persona scores into overall per-platform score
def aggregate_scores_node(state: ContentState):
    state.current_node = "aggregate_scores_node"
    state.aggregated_scores = {}
    for platform, persona_feedback in state.persona_scores.items():
        scores = [f["score"] for f in persona_feedback.values()]
        avg_score = sum(scores) / len(scores)
        state.aggregated_scores[platform] = avg_score
    return state


# 4. Check if refinement is needed
def check_refinement_node(state: ContentState):
    state.current_node = "check_refinement_node"
    # If ANY platform has low score and not max refinements
    for platform, avg_score in state.aggregated_scores.items():
        if avg_score < THRESHOLD_SCORE and state.refinement_loops[platform] < MAX_REFINEMENTS:
            return "refine"
    return "finalize"


# 5. Refine content based on feedback
def refine_content_node(state: ContentState):
    state.current_node = "refine_content_node"
    #Refines content for platforms that scored below threshold based on persona feedback.
    threshold = 8  # Minimum acceptable score

    for platform, avg_score in state.aggregated_scores.items():
        if avg_score < THRESHOLD_SCORE and state.refinement_loops[platform] < MAX_REFINEMENTS:
            # Combine feedback
            combined_feedback = "\n".join(
                f"{persona}: {data['feedback']}"
                for persona, data in state.persona_scores[platform].items()
            )

            # Call LLM for refined content
            refined_content=refine_content(state, platform, combined_feedback).content
            json_content = json.loads(refined_content)


            # Update state
            state.generated_content[platform] = json_content
            state.refinement_history[platform].append(json_content)
            state.refinement_loops[platform] += 1

    return state


# 6. Finalize content
def finalize_content_node(state: ContentState):
    state.current_node = "finalize_content_node"
    state.final_content = state.generated_content.copy()
    return state



# ---------------------------
# GRAPH SETUP
# ---------------------------

# Create graph
graph = StateGraph(ContentState)

# Add nodes
graph.add_node("generate_content", generate_content_node)
graph.add_node("score_personas", score_personas_node)
graph.add_node("aggregate_scores", aggregate_scores_node)
#graph.add_node("check_refinement", check_refinement_node)
graph.add_node("refine_content", refine_content_node)
graph.add_node("finalize_content", finalize_content_node)

# Edges: main flow
graph.add_edge("generate_content", "score_personas")
graph.add_edge("score_personas", "aggregate_scores")
#graph.add_edge("aggregate_scores", "check_refinement")

# Conditional refinement
graph.add_conditional_edges(
    "aggregate_scores",
    check_refinement_node,  # or your lambda
    {
        "refine": "refine_content",
        "finalize": "finalize_content"
    }
)

# After refinement -> back to scoring
graph.add_edge("refine_content", "score_personas")

# Final edge
graph.add_edge("finalize_content", END)

graph.set_entry_point("generate_content")

social_content_graph = graph.compile()
