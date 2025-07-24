import streamlit as st
import json

from agent import contentGraph
from src.agent.contentGraph import contentGraph as contentGraph
from tools.insight_topics import get_market_insight_topics

# --- New Market Insights Topics Section ---
def market_insights_topics_section( model, personality, temperature, top_p):
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
        with st.spinner("Generating tweet variants..."):
            result = contentGraph.invoke({
                "theme": user_input,
                "personality": personality.lower() if personality else "smart-casual",
                "model_choice": model,
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
    return selected_topic

# --- Old Trending Section (now disconnected from use) ---
def trending_section(fetch_trending, selected_theme, temperature, top_p, personality, model_choice, contentGraph):
    st.markdown("#### 🔥 Trending Topics (DISCONNECTED)")
    trending_titles = fetch_trending()
    selected = st.radio("Use a trending topic?", trending_titles, index=None, key="trending_radio", disabled=True)
    send_trending = st.button("➡️", key="send_trending", disabled=True, help="Send selected trending topic (disabled)")
    st.markdown("---")
    return selected
