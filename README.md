````markdown
# Social Media Generator (AI-Powered) 📝⚡

One-click social copy for marketers, founders, and creators.  
Generate platform-optimised captions or posts for **Twitter/X, Instagram, LinkedIn, TikTok, and Facebook**.  

- **Default provider:** Ollama (local & free)  
- **Optional provider:** OpenAI (paid)  

---

## Table of Contents
- [Built By](#built-by)  
- [Streamlit Community Cloud](#streamlit-community-cloud)  
- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Architecture](#architecture)  
- [Components](#components)  
- [Installation & Setup](#installation--setup)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [How It Works](#how-it-works)  
- [Platform-Specific Adjustments](#platform-specific-adjustments)  
- [Testing & QA](#testing--qa)  
- [Performance & Costs](#performance--costs)  
- [Security](#security)  
- [Contributing](#contributing)  
- [License](#license)  
- [FAQ](#faq)  
- [Future Roadmap](#future-roadmap)  
- [Quick Start](#quick-start)  

---

## Built By
**Mohamed Almehayla**  
- GitHub: [@malmehayla](https://github.com/malmehayla)  
- LinkedIn: [Mohamed Almehayla](https://linkedin.com/in/malmehayla)  

---

## Streamlit Community Cloud
This application is hosted on **Streamlit Community Cloud**:  
👉 [Launch the App](https://socialmediagenerator-fzxysxck3rlu9zt6etpcnm.streamlit.app)  

### Important Notes
- On the cloud version, only **OpenAI features** are available (toggle the switch).  
- You will need an **OpenAI API Key**.  

Generate your API key here: [OpenAI API Keys](https://platform.openai.com/api-keys)  

---

## Overview
The Social Media Generator is a **Python + Streamlit web app** that transforms a plain product or campaign description into **seven diverse, platform-optimised outputs**:

- **Captions:** short, catchy, emoji-rich  
- **Posts:** long-form (LinkedIn/Facebook/Instagram, 400+ words)  

The app auto-adjusts tone, structure, and length, ensures platform-specific rules (e.g., Twitter/X 280-character cap), and provides copy-to-clipboard buttons for each item.  

### Who It’s For
- Marketers and social media managers  
- Small business owners needing quick, on-brand caption banks  
- Creators and teams requiring consistent, polished content  

---

## Features
- Platform support: Twitter/X, Instagram, LinkedIn, TikTok, Facebook  
- Two providers:  
  - **Ollama (local & free)** – default; manage models in-app  
  - **OpenAI (paid)** – toggle on; model picker with fallbacks  
- Content types: Caption or Post (400+ words on supported platforms)  
- Word target control: auto-set per platform/type, enforces 80–100% window  
- JSON-shaped outputs: exactly 7 items for automation  
- Styled copy: 2–3 emojis per item + CTA ending  
- Strong UX: copy buttons, truncation/refine warnings, retry on rate limits  

---

## Tech Stack
| Component         | Version | Purpose                                         |
| ----------------- | ------: | ----------------------------------------------- |
| Python            |   3.10+ | Language/runtime                                |
| Streamlit         | 1.31.0+ | Web UI                                          |
| OpenAI Python SDK | 1.40.0+ | GPT chat completions                            |
| Requests          | 2.31.0+ | HTTP client (Ollama REST)                       |
| Ollama (binary)   |  latest | Local LLMs (`http://localhost:11434`)           |
| Git               |  latest | Version control / CI/CD (optional)              |

---

## Architecture
**High-Level Diagram:** _To be added_  

---

## Components
- **Prompt Template**  
  F-string with description, platform, tone, content type, and word target. Uses few-shot examples, `temperature=0.85`, and enforces JSON output.  

- **Provider Layer**  
  - Ollama: `/api/chat` POST with model name  
  - OpenAI: `chat.completions.create(...)`  

- **Post-Processing**  
  - Parse/repair JSON  
  - Enforce Twitter 280-char limit  
  - Apply 80–100% word window  

- **UI Layer**  
  - Sidebar inputs (provider, model/key, description, tone, etc.)  
  - Expanders with results, counts, and copy buttons  

---

## Installation & Setup
Works on **Windows/macOS/Linux**. Recommended: Conda + Python 3.10.

### 1. Clone the repo
```bash
git clone https://github.com/malmehayla/social_media_generator.git
cd social_media_generator
````

### 2. Create & activate Conda environment

**Option A (recommended) — `environment.yml`**

```bash
conda env create -f environment.yml
conda activate smg
```

**Option B — manual**

```bash
conda create -n smg python=3.10 -y
conda activate smg
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install & run Ollama

* Windows: `winget install Ollama.Ollama`
* macOS: `brew install ollama`
* Linux: see [Ollama Docs](https://ollama.com)

Pull a model:

```bash
ollama pull llama3.2:3b
# Other good options:
# qwen2.5:3b-instruct
# llama3.1:8b
```

Verify:

```bash
curl http://localhost:11434/api/version
```

### 4. (Optional) Set OpenAI API key

**macOS/Linux:**

```bash
export OPENAI_API_KEY="sk-..."
```

**Windows PowerShell:**

```powershell
setx OPENAI_API_KEY "sk-..."
```

Or in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

### 5. Verify setup

```bash
streamlit hello
python -c "import streamlit, requests, openai; print('OK')"
ollama run llama3.2:3b "Say OK in one word"
```

---

## Configuration

* **Provider toggle:** Ollama (default) or OpenAI
* **Ollama settings:** base URL, models, pull field, timeouts
* **OpenAI settings:** API key input, model dropdown (with fallbacks)
* **Word target:** auto-set per platform/type; enforced at 80–100%

---

## Usage

Run:

```bash
streamlit run app.py
```

### Sidebar Inputs

* Provider (Ollama/OpenAI)
* Product or campaign description
* Platform: Twitter/X, Instagram, LinkedIn, TikTok, Facebook
* Content type: Caption or Post
* Tone: Fun, professional, inspirational, or none
* Target word count (auto-set, editable)

### Output

* 7 items with word/char counts
* Expanders with text
* Copy buttons
* Warnings for trimming or refinement

**Example:**
Description: *Eco-friendly running shoes for urban athletes*
Platform: Instagram | Type: Caption | Tone: Fun

Result: 7 short, emoji-rich captions with varied hooks and CTAs.

---

## How It Works

1. **Input:** user provides description + settings
2. **Prompt:** app builds JSON-returning template
3. **Provider:** sends to Ollama or OpenAI
4. **Parsing & Rules:** JSON repair, word/char checks
5. **Render:** results with counts and copy buttons

---

## Platform-Specific Adjustments

| Platform  | Tweaks                                          |
| --------- | ----------------------------------------------- |
| Twitter/X | ≤ 280 chars; concise hooks dominate word target |
| Instagram | Visual/emoji hooks; “Pair with photo” prompts   |
| LinkedIn  | Professional tone; networking phrasing          |
| TikTok    | Playful, trend-aware; “Watch till the end” cues |
| Facebook  | Conversational, community-oriented              |

---

## Testing & QA

Unit tests recommended (see `app_test.py`).

| Case              | Input                  | Expected Outcome                           |
| ----------------- | ---------------------- | ------------------------------------------ |
| Empty description | `""`                   | Warning: “Description is too short”        |
| Missing API key   | OpenAI enabled, no key | Error: “Please enter your OpenAI API key.” |
| Rate limit        | High volume            | Retries with backoff, then error           |
| Non-JSON          | Provider drift         | Show error, suggest alternative model      |
| Over target words | Long item              | Trimmed with warning badge                 |
| Under 80%         | Very short             | Refine pass (non-Twitter)                  |

---

## Performance & Costs

* **Ollama:** free, local, hardware-dependent
* **OpenAI:** pay-per-token; one 7-item generation is low-cost
* Caching avoids redundant generations

---

## Security

* Do not commit secrets.
* Use `.streamlit/secrets.toml` or environment variables.
* Follow your organisation’s key management policies.

---

## Contributing

1. Fork the repo
2. Create a branch:

   ```bash
   git checkout -b feat/improve-word-refine
   ```
3. Commit:

   ```bash
   git commit -m "feat: better word-range refinement"
   ```
4. Push:

   ```bash
   git push origin feat/improve-word-refine
   ```
5. Open a Pull Request with context and testing notes

---

## License

MIT License — free to use, modify, and distribute with attribution. See `LICENSE`.

---

## FAQ

**Can I use this offline?**

* Ollama: Yes, if you pull a local model.
* OpenAI: No, requires internet.

**Why 7 items?**

* Balanced variety; changeable in code.

**Why are some Twitter/X items short?**

* 280-character limit overrides word targets.

**Why do I see a 404 on some OpenAI models?**

* Your key/org may lack access. Use `gpt-4o-mini` or another available model.

**Does the app store my API key?**

* No. Keys are runtime only.

**Can I add hashtags/mentions?**

* Yes, edit the prompt template.

---

## Future Roadmap

* Multi-language support
* Image suggestions/generation
* A/B variants + CSV export
* Analytics & UTM scaffolding
* Session history & favourites
* Scheduler integrations
* Plugin system for tone/style guides

---

## Quick Start

```bash
# Create environment
conda env create -f environment.yml
conda activate smg
conda init

# Pull local model
ollama pull llama3.2:3b

# Verify setup
streamlit hello
python -c "import streamlit, requests, openai; print('OK')"
ollama run llama3.2:3b "Say OK in one word"

# Run app
streamlit run app.py
```

Exit Streamlit with `Ctrl+C`.