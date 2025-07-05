import streamlit as st

def sidebar_settings():
    st.header("⚙️ Agent Settings")
    personality = st.selectbox("Personality", [ "Concise", "Friendly", "Formal"])
    model_choice = st.selectbox("LLM", ["OpenAI GPT-4", "Gemini 1.5 Pro"])
    st.markdown("### Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05)
    with st.expander("🆘 Help Guide"):
        st.markdown("""
            ### 💡 What this Agent Can Do

            - 🔍 **Fetch trending topics** from the web to inspire your content
            - 🧠 **Generate multiple tweet & LinkedIn variants** for your selected theme
            - 🤖 **Score and select the best idea** automatically using GPT 3 turbo as judge
            - ♻️ **Retry generation** until a high-quality result is found
            - 🗃️ **Save top-performing posts** to a vector database
            - 🧭 **Retrieve examples from memory** to guide future outputs
            - 🎛️ **Customize model behavior** (LLM choice, temperature, tone)
            - 💬 **Chat-like interface** with full conversation history
            - 📊 **See token usage and cost** during each session

            ### ✅ How to Use

            1. **Pick a topic** or use the trending suggestions
            2. **Click 'Generate'** to create post variants
            3. **Review the output** in the tweet display area
            4. **Use 'Retry'** if you're not satisfied
            5. **Post to X or LinkedIn** — ready to copy!
                """)
    return personality, model_choice, temperature, top_p
