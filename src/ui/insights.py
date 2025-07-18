import streamlit as st
import os
from typing import Optional
from tavily import TavilyClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
import pypdf
from src.agent.businessinsightsGraph import insight_graph as businessInsightsGraph

business_store = Chroma(
        collection_name="business_info",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )

def extract_website_content(url: str) -> Optional[str]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        st.error("Tavily API key not found in environment.")
        return None
    try:
        client = TavilyClient(api_key=api_key)
        response = client.extract(url)
        results = response.get("results", [])
        if results and "raw_content" in results[0]:
            return results[0]["raw_content"]
        else:
            st.warning("No content extracted from the website.")
            return None
    except Exception as e:
        st.error(f"Failed to extract website content: {e}")
        return None

# def chunk_and_store_business_info(text: str, source: str, metatags: dict):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_text(text)
#     docs = [Document(page_content=chunk, metadata={**metatags, "source": source}) for chunk in chunks]
#     business_store.add_documents(docs)
#     return len(docs)

# def generate_research_questions(summary: dict) -> list:
#     industry = summary.get("Industry / Niche", "the business")
#     product = summary.get("Product or Service Description", "the product")
#     target = summary.get("Target Audience", "customers")

#     return [
#         f"What are common pain points for {target} in the {industry} industry?",
#         f"What market trends are currently shaping the {industry} sector?",
#         f"What competing solutions exist for {product}?",
#         f"What kind of content engages {target} in the {industry} space?",
#         f"What feedback or issues do users have with similar products to {product}?",
#         f"What are misconceptions or hesitations {target} might have regarding {product}?",
#         f"What unique angles could be used to position {product} in the market?"
#     ]

# def search_tavily(query: str, api_key: str, site: str = None):

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "query": query,
#         "include_answer": True,
#         "max_results": 5,
#         "search_depth": "advanced"
#     }

#     if site:
#         payload["include_domains"] = [site]

#     response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
#     if response.status_code == 200:
#         return response.json().get("answer", "")
#     return ""

# def gather_all_insights(business_description: str, api_key: str):
#     research_questions = generate_research_questions(business_description)
#     platforms = ["reddit.com", "quora.com", "twitter.com", "news.ycombinator.com"]
    
#     all_answers = []

#     for question in research_questions:
#         for platform in platforms:
#             result = search_tavily(question, api_key, site=platform)
#             if result:
#                 all_answers.append(f"From {platform} on '{question}':\n{result}\n")
    
#     return all_answers



# def summarize_insights(all_answers: list, openai_api_key: str):
#     chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)

#     text_blob = "\n".join(all_answers)[:12000]  # Avoid context limits
#     prompt = ChatPromptTemplate.from_template("""
#     Based on the following social media discussions and answers, summarize key insights for:
#     - Market trends
#     - Customer pain points
#     - Competitive landscape
#     - Content strategy suggestions

#     --- Begin Data ---
#     {text_blob}
#     --- End Data ---
#     """)

#     chain = prompt | chat
#     result = chain.invoke({"text_blob": text_blob})
#     return result.content

# def summarize_business_input(business_input: str, openai_api_key: str) -> dict:
#     chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)

#     prompt = ChatPromptTemplate.from_template("""
#     You are a startup analyst. Analyze the following business information and extract a structured summary:

#     - Industry / Niche
#     - Target Audience
#     - Product or Service Description
#     - Problem it Solves
#     - Key Features or Keywords
#     - Unique Selling Points (USP)

#     Respond in JSON format.

#     --- Business Info ---
#     {text}
#     """)

#     chain = prompt | chat
#     result = chain.invoke({"text": business_input[:12000]})  # Truncate if too long
#     try:
#         import json
#         return json.loads(result.content)
#     except json.JSONDecodeError:
#         return {"error": "Could not parse summary"}
    

def market_insights_tab(personality, model_choice, temperature, top_p):
    st.markdown("Upload your business info to generate data-driven insights for better content.")

    website_url = st.text_input("Website or About Page URL")
    uploaded_file = st.file_uploader("Upload a Pitch Deck or Business Description (PDF or TXT)", type=["pdf", "txt"])

    business_input = ""
    source = None
    metatags = {}

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            # Extract text from PDF
            reader = pypdf.PdfReader(uploaded_file)
            business_input = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            # Assume plain text
            business_input = uploaded_file.read().decode("utf-8", errors="ignore")
        source = "file"
        metatags["filename"] = uploaded_file.name
    if website_url:
        with st.spinner("Extracting content from website via Tavily..."):
            website_content = extract_website_content(website_url)
        if website_content:
            if business_input:
                business_input += "\n\n" + website_content
                source = "file+website"
            else:
                business_input = website_content
                source = "website"
            metatags["website_url"] = website_url

    if business_input:
        if st.button("Uplod Business Info", key="embed_business_info"):
            with st.spinner("Gathering and analyzing insights from social platforms.."):
                result = businessInsightsGraph.invoke({
                "business_input": business_input,
                "personality": personality.lower() if personality else "smart-casual",
                "model_choice": model_choice,
                "temperature": temperature,
                "top_p": top_p
            })
            summarized_insights = result.get("summarized_insights", "No insights found.")
            if summarized_insights:
                st.divider()
                st.subheader("📡 Social Platform Market Insights")
                if isinstance(summarized_insights, dict):
                    for category, insight in summarized_insights.items():
                        st.markdown(f"**{category}**")
                        # If insight is a list, print each as a bullet
                        if isinstance(insight, list):
                            for item in insight:
                                st.markdown(f"- {item}")
                        else:
                            # If it's a string, split by comma and list as bullets
                            items = [i.strip() for i in str(insight).split(", ") if i.strip()]
                            for item in items:
                                st.markdown(f"- {item}")
                        st.markdown("")  # Add a blank line for spacing
                else:
                    st.markdown(str(summarized_insights))

            #     num_chunks = chunk_and_store_business_info(business_input, source, metatags)
            # st.success(f"Stored {num_chunks} chunks in the vector database.")

            # st.divider()
            # st.subheader("📡 Social Platform Market Insights")

            # with st.spinner("Summarizing business input..."):
            #     business_summary = summarize_business_input(
            #         business_input,
            #         openai_api_key=os.getenv("OPENAI_API_KEY")
            #     )

            # with st.spinner("Analyzing real discussion data from Twitter, Reddit, Quora and YCombinator..."):
            #     answers = gather_all_insights(business_summary, api_key=os.getenv("TAVILY_API_KEY"))
            #     insights_summary = summarize_insights(answers, openai_api_key=os.getenv("OPENAI_API_KEY"))

            # if insights_summary:
            #     st.markdown(insights_summary)
    else:
        st.info("Please enter a website URL or upload a business file to begin.") 
        st.stop()

    