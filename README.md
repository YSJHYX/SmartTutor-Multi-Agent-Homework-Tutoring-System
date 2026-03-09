# SmartTutor: Multi-Agent Homework Tutoring System 🤖📚



## Overview
SmartTutor is a highly reliable, multi-turn AI homework tutoring agent designed for Math and History subjects. Built on a multi-agent routing architecture, the system prioritizes **strict LLM guardrails** to prevent prompt injection and ensure the agent stays strictly on academic topics. 

This project was developed as part of the CSIT5900 course at Hong Kong University of Science and Technology (HKUST).

## ✨ Key Features
* **Multi-Agent Routing (Triage):** Dynamically routes user queries to specialized agents (Math Tutor, History Tutor, or Summary Agent) based on semantic intent.
* **Robust Guardrails (Anti-Spoofing):** Implements strict validation to reject non-homework queries (e.g., travel logistics, dangerous hypotheticals) with context-aware, personalized rejection messages. 
* **Persona Adaptation:** Automatically captures user identities (e.g., "Primary school student" or "University Year 1") and adjusts the depth and tone of explanations accordingly, including out-of-syllabus warnings.
* **Top-K Memory Module:** Maintains conversational context using a sliding window approach, enabling multi-turn dialogue and accurate conversation summarization.
* **Hybrid Model Architecture:** Uses **DeepSeek** for advanced mathematical reasoning and **Azure OpenAI (gpt-4o-mini)** for history, routing, and guardrail checks.

## 🛠️ Tech Stack
* **Framework:** `openai-agents`, `pydantic`, `asyncio`
* **LLMs:** DeepSeek API (`deepseek-chat`), Azure OpenAI (`gpt-4o-mini`)

## 🚀 Getting Started

### Prerequisites
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install openai-agents pydantic python-dotenv
