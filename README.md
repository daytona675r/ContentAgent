# AI Social Content Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![LangGraph](https://img.shields.io/badge/AI-Agents%20via%20LangGraph-7A3E9D)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered agentic application for **multi-platform content ideation, generation, scoring, and refinement**.  
The app uses **LangGraph** to orchestrate two main processes:  
1. **Business Insights Graph** – Processes uploaded business information and market data.  
2. **Social Content Graph** – Generates and optimizes social media posts for multiple platforms.

---

## Features

- Upload **business PDFs or text** to extract insights and summarize them.
- Generate **customized market insights** from social platform data (e.g., Twitter, LinkedIn).
- Create **platform-specific content**:
  - Twitter (short & punchy)
  - Facebook (community-oriented)
  - LinkedIn (professional/thought leadership)
  - Instagram (visual/meme-oriented)
- **Multi-persona LLM-as-a-Judge scoring** (Strategist, Customer, Expert, Investor).
- **Iterative refinement loop** until a quality threshold is reached.
- Summarized **overview tab** + detailed per-platform tabs.
- Full **real-time status updates** during graph execution.

---

## Architecture

### 1. Business Insights Graph
- **Input:** Uploaded business documents or text.
- **Steps:**
  1. Extract and embed raw data into vector DB.
  2. Summarize insights for later use in content generation.
  3. Store summarized insights for querying.

**Purpose:** Provides contextual knowledge for the Social Content Graph.

### 2. Social Content Graph
- **Input:** Selected topic or user input + summarized insights.
- **Steps:**
  1. Generate 4 platform-specific posts in a single LLM call (JSON output).
  2. Score each post via 4 personas (16 judgments total).
  3. Aggregate scores; trigger refinement loop if below threshold.
  4. Store refined drafts in history and finalize best variant.
  5. Display summary + detailed breakdown per platform.

---

## Installation

```bash
git clone <repo-url>
cd ai-social-content-generator
pip install -r requirements.txt
```

**Environment variables (set in `.env`):**
```
OPENAI_API_KEY=your-key
```

---

## Running the App

```bash
streamlit run app.py
```

---

## Usage Guide

1. **Upload Business Info**  
   Go to the sidebar and upload PDFs or text describing your company.  
   The system summarizes and stores this in the vector DB.

2. **Generate Market Insights**  
   Trigger the Insights Graph to analyze trends and store structured insights.

3. **Select or Input Topic**  
   Pick from generated topics or enter your own.

4. **Generate Content**  
   The Social Content Graph creates posts for Twitter, Facebook, LinkedIn, and Instagram.

5. **Scoring & Refinement**  
   Multi-persona judging loop refines until acceptable scores are reached.

6. **View Results**  
   Summary tab gives an overview; detailed tabs show full drafts and history.

---

## Badges & Tech

- **LLM Orchestration:** LangGraph
- **Frontend:** Streamlit
- **Storage:** Vector DB (e.g., Chroma/FAISS)
- **Evaluation:** LLM-as-a-Judge with multi-persona perspectives

---

## License

MIT License. See LICENSE for details.
