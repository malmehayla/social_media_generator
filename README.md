# Social Media Generator

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-ff4b4b)
![OpenAI](https://img.shields.io/badge/OpenAI-API-00a37d)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-2b90d9)

Generate platform-optimised **captions** or **posts** for Twitter/X, Instagram, LinkedIn, TikTok, and Facebook with strict word targets and platform rules (e.g., X 280-char cap).  
- **Default provider:** **Ollama** (free, local LLMs)  
- **Optional provider:** **OpenAI** (paid) — toggle and model picker in UI  
- **Word target enforcement:** 80–100% band, with an auto-refine pass (non-X platforms)

---

## ✨ Features

- Platforms: **Twitter/X**, **Instagram**, **LinkedIn**, **TikTok**, **Facebook**
- Content types: **Caption** (short) or **Post** (long, 400+ words on supported platforms)
- **Strict word target control** (80–100% of your target) + auto-refine when too short
- Platform rules: **Twitter/X is truncated to 280 characters**
- Tone presets (fun, professional, inspirational)
- Copy-to-clipboard buttons for each generated item
- **Ollama integration**: list installed models, refresh, and **pull models** directly in the app
- **OpenAI integration**: model dropdown (with safe fallbacks), key input via sidebar

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **LLMs**: Ollama local models (default), OpenAI Chat Completions (optional)
- **Language**: Python 3.10
- **HTTP**: `requests` (for Ollama REST)

---

## 📁 Project Structure

