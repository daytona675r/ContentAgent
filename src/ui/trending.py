import streamlit as st

def trending_section(fetch_trending, selected_theme, temperature, top_p, personality, model_choice, contentGraph):
    st.markdown("#### 🔥 Trending Topics")
    trending_titles = fetch_trending()
    selected = st.radio("Use a trending topic?", trending_titles, index=None)
    send_trending = st.button("➡️", key="send_trending", disabled=selected is None, help="Send selected trending topic")
    st.markdown("---")
    if send_trending and selected:
        user_input = selected
        st.session_state.last_input = user_input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Generating tweet variants..."):
            result = contentGraph.invoke({
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
    return selected
