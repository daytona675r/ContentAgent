import streamlit as st
import json

from agent.social_content_graph import social_content_graph
from agent.social_content_graph import ContentState
from tools.business_info import get_business_info
from tools.insights import get_market_insight_topics, get_market_insights
from ui.content_worker import run_content_generation

# --- New Market Insights Topics Section ---
def market_insights_topics_section( model, personality, temperature, top_p):
    if st.session_state.get("reset_market_topic", False):
        st.session_state["selected_market_topic"] = None
        st.session_state["input"] = "" 
        st.session_state["reset_market_topic"] = False
        
    # Load topics only once
    if "market_insight_topics" not in st.session_state:
            with st.spinner("Creating topics from business insights..."):
                st.session_state.market_insight_topics = get_market_insight_topics(model, personality, temperature, top_p)

    # Create columns for title + optional button
    col1, col2 = st.columns(2)  # adjust ratio for spacing

    with col1:
        st.markdown("#### 📊 Market Insights Topics")

    with col2:
        # Show button only if topics are already loaded
        if "market_insight_topics" in st.session_state:
            if st.button("🔁", key="retryInsightTopics", help="Recreate topics from insights"):
                with st.spinner("Recreating..."):
                    st.session_state.market_insight_topics = get_market_insight_topics(
                        model, personality, temperature, top_p
                    )

    # Use a persistent key for the radio box
    selected_topic = st.radio(
        "Choose a topic to discuss:",
        st.session_state.market_insight_topics,
        index=None,
        key="selected_market_topic"
    )


    # If a topic is selected, set it as the chat input value
    send_topic = st.button("➡️", key="send_trending", disabled=selected_topic is None, help="Send selected topic")
    st.markdown("---")
    if send_topic and selected_topic:
        user_input = selected_topic
        st.session_state.last_input = user_input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        final_state=run_content_generation(user_input)

        if final_state:
            st.session_state.final_state = final_state
            st.session_state.chat_history.append({"role": "agent", "content": "Your content is generated successfully! Look at the summary tab for details."})
           
            st.session_state["reset_market_topic"] = True
        st.rerun()
    return selected_topic

# --- Old Trending Section (now disconnected from use) ---
def trending_section(fetch_trending, selected_theme, temperature, top_p, personality, model_choice, contentGraph):
    st.markdown("#### 🔥 Trending Topics (DISCONNECTED)")
    trending_titles = fetch_trending()
    selected = st.radio("Use a trending topic?", trending_titles, index=None, key="trending_radio", disabled=True)
    send_trending = st.button("➡️", key="send_trending", disabled=True, help="Send selected trending topic (disabled)")
    st.markdown("---")
    return selected
