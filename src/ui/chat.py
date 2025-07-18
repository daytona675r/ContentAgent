import streamlit as st

def chat_input_area(selected_theme):
    placeholder = selected_theme if selected_theme else "Enter a content theme (e.g. AI for solopreneurs)..."
    return st.chat_input(placeholder)

def chat_history_area():
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "agent"
                and idx == len(st.session_state.chat_history) - 1
                and hasattr(st.session_state, "final_state")
            ):
                st.metric(label="🔥 Score", value=round(st.session_state.final_state["score"], 2), delta="🎯 Target: 0.75")
                if hasattr(st.session_state, "last_linkedin_variant") and st.session_state.last_linkedin_variant:
                    with st.expander("Show LinkedIn Variant"):
                        st.markdown(st.session_state.last_linkedin_variant)

def retry_button(personality, model_choice, temperature, top_p, contentGraph):
    if st.session_state.last_input and st.session_state.chat_history:
        last_agent_index = next(
            (i for i in reversed(range(len(st.session_state.chat_history)))
             if st.session_state.chat_history[i]["role"] == "agent"),
            None
        )
        if last_agent_index is not None:
            if st.button("🔁", key="retry", help="Retry with same theme"):
                with st.spinner("Retrying..."):
                    result = contentGraph.invoke({
                        "theme": st.session_state.last_input,
                        "personality": personality.lower() if personality else "smart-casual",
                        "model_choice": model_choice,
                        "temperature": temperature,
                        "top_p": top_p
                    })
                final_tweet = result.get("selected_idea", "No tweet generated.")
                score = result.get("score", 0.0)
                updated_reply = f"Here's your best tweet variant:\n\n> {final_tweet}\n\n(Score: {score:.2f})"
                st.session_state.chat_history[last_agent_index] = {
                    "role": "agent",
                    "content": updated_reply
                }
                st.rerun()
