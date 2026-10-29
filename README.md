# PULSE

**A LangGraph workflow for generating, evaluating, and refining platform-specific content.**

**[Try the live demo](https://pulse-demo-web.onrender.com)** · [Source code](https://github.com/daytona675r/ContentAgent) · [Hosted smoke tests](https://github.com/daytona675r/ContentAgent/actions/workflows/hosted-smoke.yml)

PULSE explores a practical engineering question: **how do you turn variable AI-generated drafts into an inspectable workflow with explicit evaluation and revision steps?**

Given a business brief, the workflow produces posts for X/Twitter, LinkedIn, Facebook, and Instagram; evaluates each draft; revises below-threshold results; and retains the before/after trace.

## Explore the demo

Choose one of three **fictional** briefs:

- **Launchpath:** a B2B onboarding workspace for operations teams
- **Switchback:** a repairable commuter light
- **Night Market North:** a neighborhood evening market

For each platform, inspect the **initial draft → strategist feedback and score → revised post → accepted score**. You can see what changed and why, rather than receiving only a final answer.

> **What is real vs. simulated?** The public demo runs the actual FastAPI → LangGraph orchestration and evaluation/refinement branches. The `ReplayProvider` supplies **pre-authored, deterministic** example posts, feedback, and scores. No live LLM generates or judges these posts; displayed token/cost values are illustrative. No OpenAI, Google, or Tavily credentials are needed.

## Core capabilities — beyond the public demo

The original PULSE application combines **two LangGraph workflows**: one to establish business and market context, and one to turn that context into evaluated social content. These broader capabilities are implemented in the original Streamlit/model-backed code; **the public React demo intentionally exercises only the Social Content Graph with fictional inputs and replayed provider responses**.

### 1. Business Insights Graph — build the context

The [Business Insights Graph](src/agent/businessinsightsGraph.py) supports the research stage of the original application:

1. **Ingest business information** from a pitch deck or business description (PDF/text), or content extracted from a website in the [Streamlit insights UI](src/ui/insights.py).
2. **Extract structured business context** with an LLM: product, target audience, customer problem, industry, and positioning.
3. **Generate research questions** around customer pain points, trends, competitors, and potential content angles.
4. **Gather and synthesize external insights** using Tavily searches scoped to Reddit, Quora, X/Twitter, and Hacker News, then summarize the findings with an LLM.
5. **Store business input and insight summaries in Chroma** for retrieval and reuse when developing content topics.

The purpose is to ground downstream writing in a specific business and audience, rather than generating generic marketing copy. This research workflow requires external API credentials and is **not run by the public demo**.

### 2. Social Content Graph — generate, judge, refine

The [Social Content Graph](src/agent/social_content_graph.py) turns a selected topic and available business insights into distinct drafts for **X/Twitter, LinkedIn, Facebook, and Instagram**. It then validates structured provider responses, evaluates each platform's draft, aggregates evaluation scores, and conditionally refines low-scoring content. Revised posts are **re-evaluated**, with drafts, scores, feedback, and accepted outputs kept in the graph state.

In the **model-backed implementation**, the [evaluation prompt](src/agent/prompts.py) asks an LLM to judge the content from **four complementary personas**:

| Evaluator | What it examines |
| --- | --- |
| **Social Media Strategist** | Hook, clarity, engagement potential, platform fit, brand alignment |
| **Target Customer** | Relevance to pain points, clarity of value, usefulness |
| **Industry Expert** | Domain accuracy, credibility, quality of insights |
| **Investor** | Business relevance, professionalism, market potential |

Each perspective is intended to return a score and actionable feedback; the graph can aggregate those scores and use the feedback to guide refinement. The current graph limits retries to **one refinement pass per platform**.

**Demo scope:** the deployed `ReplayProvider` supplies **one fictional strategist evaluation per platform**, with deterministic draft → feedback → revised post → accepted score. The four-persona, live-LLM judging path and Business Insights Graph remain in the original codebase; they are **not verified by the public hosted demonstration**.

## Architecture

**React + TypeScript (Vite)** → **FastAPI** → **LangGraph** → **pluggable provider**

```mermaid
flowchart TD
    B[Business brief] --> G[Generate four platform drafts]
    G --> E[Evaluate each platform]
    E --> Q{Below acceptance threshold?}
    Q -->|Yes| R[Refine using feedback]
    R --> E
    Q -->|No| A[Accept content and preserve trace]
```

The same graph runs with an injected provider. The public demo substitutes authored responses at the provider boundary instead of replacing graph execution with a static API response.

**Implementation details worth inspecting:**

- **Graph and gates:** [`social_content_graph.py`](src/agent/social_content_graph.py) coordinates generation, scoring, conditional refinement, re-scoring, and finalization.
- **Provider boundary:** [`demo_provider.py`](src/agent/demo_provider.py) supplies repeatable scenario-specific responses without external keys.
- **API contracts:** [`api/main.py`](api/main.py) exposes typed scenario/result endpoints, validates scenario IDs, and reports workflow failures explicitly.
- **Client:** [`frontend/src`](frontend/src) renders per-platform draft, critique, refinement, and accepted score.
- **Verification:** tests cover contracts, graph flow, frontend rendering, and the deployed API/frontend integration.

The hosted example uses **one simulated strategist evaluator**. Multi-persona LLM prompts and business-insight experiments remain in the older codebase; they are **not** demonstrated as live AI features in the public deployment.

## Run locally

Requires **Python 3.11–3.13** and **Node.js 24** (the version used by CI).

**Terminal 1 — API**

```bash
git clone https://github.com/daytona675r/ContentAgent.git
cd ContentAgent
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend**

```bash
cd ContentAgent/frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The frontend defaults to `http://localhost:8000` for local API calls. Both services work in deterministic demo mode without provider secrets.

## Deployment and verification

- **Frontend:** [pulse-demo-web.onrender.com](https://pulse-demo-web.onrender.com)
- **API:** [pulse-demo-api-iwj4.onrender.com](https://pulse-demo-api-iwj4.onrender.com) ([health](https://pulse-demo-api-iwj4.onrender.com/health))
- **Render configuration:** [`render.yaml`](render.yaml) documents the backend and static frontend; the live services were provisioned separately with equivalent settings.

`VITE_API_BASE_URL` is the public API HTTPS URL **at frontend build time**. `PULSE_ALLOWED_ORIGINS` contains the allowed frontend origins. The hosted service uses `DEMO_MODE=true`; no model-provider keys are configured.

Run the public integration check:

```bash
API_URL=https://pulse-demo-api-iwj4.onrender.com \
FRONTEND_URL=https://pulse-demo-web.onrender.com \
bash scripts/deployment_smoke.sh
```

Or inspect the GitHub Actions checks: [Quality](https://github.com/daytona675r/ContentAgent/actions/workflows/quality.yml) · [Portfolio Acceptance](https://github.com/daytona675r/ContentAgent/actions/workflows/portfolio-acceptance.yml) · [Hosted Smoke](https://github.com/daytona675r/ContentAgent/actions/workflows/hosted-smoke.yml).

Render's free backend can sleep when idle, so the first request may take longer. The demo is a portfolio showcase, not an always-on production service.

<details>
<summary>Original Streamlit prototype</summary>

The original [`app.py`](app.py) includes a separate Streamlit interface and earlier business-insights/model-backed experiments. It is retained as legacy code and is **not** the React demo linked above.

Run its credential-free sample view with:

```bash
DEMO_MODE=1 streamlit run app.py
```

The older screenshots in [`static/`](static/) depict this prototype, not the public React UI.

</details>

## License

[MIT](LICENSE).
