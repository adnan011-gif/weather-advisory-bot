# Weather-Advisory Support Bot

A production-grade, zero-hallucination meteorological safety advisory assistant built with LangGraph, Open-Meteo live forecasts, and certified Standard Operating Procedures (SOPs).

---

## Core Architecture Principles

1. **Separation of Policy and Logic**: All safety rules and thresholds live in `sops/*.yaml`. Fact definitions, variable bindings, and time window resolution live in `config/facts.yaml`.
2. **SOP Freshness**: Policies and facts are dynamically reloaded from disk at the start of every turn. Adding or editing an SOP takes immediate effect without requiring a server restart or code change.
3. **Deterministic Authority**: The LLM classifies intent and rephrases advisories. Safety checks, thresholds, and conflict resolution are 100% deterministic. Clear policies and informational advisories bypass the LLM completely (`answer_source: template`, reason `no_llm_needed`).
4. **Zero Hallucination Guarantee**: All numbers and cited policy IDs in language responses are strictly validated against an allow-list before presentation.
5. **Multi-Model Quota Resilience**: Automatic retries with exponential backoff on 429/503 errors and seamless fallback across model chains (`GEMINI_MODEL` -> `GEMINI_FALLBACK_MODELS`).

---

## Local Development Setup

### 1. Environment & Dependencies
Requires **Python 3.11+**.

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and provide your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-1.5-flash,gemini-1.5-pro
```

### 3. Running the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment to Streamlit Community Cloud

### Step-by-Step Deployment Guide

1. **Push Repository to GitHub**:
   Ensure your code is pushed to your GitHub repository (e.g. `main` branch). Verify that `.streamlit/secrets.toml` and `.env` are excluded by `.gitignore`.

2. **Connect to Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set **Main file path** to `app.py`.

3. **Configure Secrets**:
   - In the app creation dashboard (or under **App settings > Secrets**), paste your configuration using the TOML format below:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = "gemini-1.5-flash,gemini-2.0-flash"
```

4. **Deploy**:
   - Click **Deploy!**. Streamlit Cloud will create a clean Python virtual environment, install the pinned dependencies from `requirements.txt`, and start the app.

---

## Running Automated Tests

Run the full offline test suite (all tests use fake clients and mocks, zero live network/LLM calls):

```bash
pytest -v
```

Static verification:
```bash
python -m compileall src scripts tests app.py
python -m pyflakes src scripts tests app.py
```
