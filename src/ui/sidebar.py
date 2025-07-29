import streamlit as st

def sidebar_settings():
    st.header("⚙️ Agent Settings")
    personality = st.selectbox("Personality", [ "Concise", "Friendly", "Formal"])
    model_choice = st.selectbox("LLM", ["OpenAI GPT-4.5", "Gemini 1.5 Pro"])
    st.markdown("### Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05)
    with st.expander("🆘 Help Guide"):
        st.markdown("""
            ### How to use this Content Generator

            ---

            #### 1. Provide business information *(important!)*
            - Upload or paste your **business details** (e.g., products, target audience, tone, goals).
            - This information is stored in the database and used to:
            - Generate **market insights** from social platforms.
            - Create **highly relevant content topics** tailored to your business.

            *(Tip: You can update this info anytime — old data will be replaced.)*

            ---

            #### 2. Generate market insights
            - The **Market Insights Graph** collects and summarizes social trends.
            - These insights are combined with your business data for smarter topic suggestions.

            ---

            #### 3. Select or enter a content topic
            - Pick a suggested topic from the insights or type your own.

            ---

            #### 4. Let the AI agent work
            - The agent creates **platform-specific posts** for:
            - **Twitter:** short & viral  
            - **LinkedIn:** thought-leadership  
            - **Facebook:** community-focused  
            - **Instagram:** meme-like & visual  

            - Posts are **scored by 4 personas** (strategist, industry expert, customer, investor) and **refined** until they meet quality standards.

            ---

            #### 5. Review the results
            - **Summary Tab:** Overview of all posts with scores and short previews.
            - **Platform Tabs:** Full post text and refinement history per platform.

            ---

            #### 6. Next steps
            - Copy posts directly for scheduling or editing.
            - Use refinement insights to guide your brand’s future content strategy.

            ---

            **Tips:**
            - The better your **business info**, the more relevant your topics and posts.
            - Market insights update periodically for trend accuracy.
            - Refinement stops once all posts pass the target quality score.

                """)
    return personality, model_choice, temperature, top_p
