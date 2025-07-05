# 📄 streamlit_app.py

import requests
import streamlit as st

from src.agent.graph import graph  # <- this is your LangGraph workflow
from langgraph.graph import StateGraph
from typing import TypedDict, List
from tools.trending import get_trending_topics
from src.ui.sidebar import sidebar_settings
from src.ui.trending import trending_section
from src.ui.chat import chat_input_area, chat_history_area, retry_button
from src.ui.token_counter import floating_token_box

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_input" not in st.session_state:
    st.session_state.last_input = ""


# --- Main App ---
st.title("🤖 AI Content Agent")

with st.sidebar:
    personality, model_choice, temperature, top_p = sidebar_settings()


@st.cache_data(ttl=3600)
def fetch_trending():
    return get_trending_topics()

selected_theme = trending_section(fetch_trending, None, temperature, top_p, personality, model_choice, graph)




# Input from user, use trending topic as placeholder if selected
user_input = chat_input_area(selected_theme)

if user_input:
    st.session_state.last_input = user_input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Generating tweet variants..."):
        result = graph.invoke({
            "theme": user_input,
            "personality": personality.lower() if personality else "smart-casual",
            "model_choice": model_choice,
            "temperature": temperature,
            "top_p": top_p
        })
    final_tweet = result.get("selected_idea", "No tweet generated.")
    linkedInVariant = result.get("linkedin_variant", "No LinkedIn variant generated.")
    agent_message = f"Here's your best tweet variant:\n\n> {final_tweet})"
    st.session_state.chat_history.append({"role": "agent", "content": agent_message})
    st.session_state.last_linkedin_variant = linkedInVariant
    st.session_state.final_state = result
    st.rerun()



# Display chat history (reverse for newest at bottom)
chat_history_area()



# --- Retry Button ---
retry_button(personality, model_choice, temperature, top_p, graph)



# Floating window for token/price info only (bottom right, 1/10 of window width)
if hasattr(st.session_state, "final_state") and st.session_state.final_state:
    floating_token_box(st.session_state.final_state)


