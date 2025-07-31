# 📄 streamlit_app.py

import requests
import streamlit as st

from agent.social_content_graph import social_content_graph
from agent.social_content_graph import ContentState
from src.agent.contentGraph import contentGraph
from langgraph.graph import StateGraph
from typing import TypedDict, List
from tools.business_info import get_business_info
from tools.insights import get_market_insights
from tools.trending import get_trending_topics
from src.ui.sidebar import sidebar_settings
from src.ui.trending import market_insights_topics_section
from src.ui.chat import chat_input_area, chat_history_area, retry_button
from src.ui.token_counter import floating_token_box
from src.ui.insights import market_insights_tab
from ui.content_worker import run_content_generation
from ui.facebook import facebook_tab
from ui.instagram import instagram_tab
from ui.linkedin import linkedin_tab
from ui.summary import summary_tab
from ui.twitter import twitter_tab

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "final_state" not in st.session_state:
    st.session_state.final_state = ContentState.__new__(ContentState)

if "final_content" not in st.session_state:
    st.session_state.final_content = []


# --- Main App ---
st.title("🌐 AI Content Agent")

with st.sidebar:
    personality, model_choice, temperature, top_p = sidebar_settings()

# @st.cache_data(ttl=3600)
# def fetch_trending():
#     return get_trending_topics()

# --- Tabs ---
tabs = st.tabs([
    "## 📊 INSIGHTS",
    "## 🗪 CHAT",
    "## 📝 SUMMARY",
    "## 𝕏 / TWITTER",
    "## LINKED[IN]",
    "## ƒACEBOOK",
    "## 🅾 INSTA"
])

with tabs[0]:  # 📊 Market Insights
    market_insights_tab(personality, model_choice, temperature, top_p)

with tabs[1]:  # Chat Agent
    selected_theme = market_insights_topics_section(model_choice,personality, temperature, top_p)
    # If a topic is selected, use it as the value for the chat input
    user_input = chat_input_area(selected_theme if selected_theme else "")

    if user_input:
        st.session_state.last_input = user_input
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        final_state=run_content_generation(user_input)
       
        if final_state:
            st.session_state.final_state = final_state
            st.session_state.chat_history.append({"role": "agent", "content": "Your content is generated successfully! Look at the summary tab for details."})
            st.session_state["input"] = ""  # Clear chat input if your input uses key="input"
            st.session_state["selected_market_topic"] = None  # Deselect the radio button
        st.rerun()

    # Display chat history (reverse for newest at bottom)
    chat_history_area()

    # --- Retry Button ---
    retry_button(personality, model_choice, temperature, top_p, contentGraph)

    #Floating window for token/price info only (bottom right, 1/10 of window width)
    if hasattr(st.session_state, "final_state") and st.session_state.final_state:
        floating_token_box(st.session_state.final_state)

with tabs[2]:  
    summary_tab()

with tabs[3]: 
    twitter_tab()

with tabs[4]: 
    linkedin_tab()

with tabs[5]: 
    facebook_tab()

with tabs[6]: 
    instagram_tab()




