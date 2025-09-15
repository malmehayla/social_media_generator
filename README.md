## Social Media Generator (AI-Powered) 📝⚡
One-click social copy for marketers, founders, and creators.
Generate platform-optimized captions or posts for Twitter/X, Instagram, LinkedIn, TikTok, and Facebook.
Default provider: Ollama (local & free). Optional provider: OpenAI (paid).

## Built By  
**Mohamed Almehayla**  
- GitHub: [@malmehayla](https://github.com/malmehayla)  
- LinkedIn: [Mohamed Almehayla](https://linkedin.com/in/malmehayla)  

## Streamlit Community Cloud  
This application is hosted on **Streamlit Community Cloud**:  
👉 [Launch the App](https://socialmediagenerator-fzxysxck3rlu9zt6etpcnm.streamlit.app)  

### Important Notes  
- On the cloud version, only **OpenAI features** are available (make sure to toggle the switch).  
- You will need an **OpenAI API Key** to use these features.  

You can generate your API key here: [OpenAI API Keys](https://platform.openai.com/api-keys)  

## Overview
Social Media Generator is a Python + Streamlit web app that turns a plain-language product or campaign description into seven diverse, platform-optimized items:
- Captions (short, hooky, emoji-rich)
- Posts (long-form for LinkedIn/Facebook/Instagram, 400+ words where applicable)

It auto-adjusts tone, structure, and length based on your selections, ensures platform-specific constraints (e.g., Twitter/X 280-char cap), and includes copy-to-clipboard buttons for each item.

Who it’s for

- Marketers and social media managers shipping copy daily
- Small business owners who want a quick, on-brand caption bank
- Creators and teams needing consistent ideas with a polished format

## Features
- Platform support: Twitter/X, Instagram, LinkedIn, TikTok, Facebook
- Two providers:
    - Ollama (local & free) — default; list installed models, pull new models in-app
    - OpenAI (paid) — toggle on; model picker with safe fallbacks (e.g., gpt-4o-mini)
- Content types: Caption (concise) or Post (structured long-form, 400+ words on supported platforms)
- Word target control: Auto-sets by platform & type; app enforces 80–100% of your chosen target with a refine pass (non-Twitter)
- JSON-shaped outputs: Exactly 7 items for easy parsing/automation
- Style baked in: 2–3 emojis per item + CTA ending (e.g., “Shop now!”, “Request a demo!”)
- Robust UX: Copy buttons per item, warnings for truncation/trim/refine, rate-limit retries

## Tech Stack
| Component         | Version | Purpose                                         |
| ----------------- | ------: | ----------------------------------------------- |
| Python            |   3.10+ | Language/runtime                                |
| Streamlit         | 1.31.0+ | Web UI                                          |
| OpenAI Python SDK | 1.40.0+ | Chat Completions for GPT models                 |
| Requests          | 2.31.0+ | HTTP client (Ollama REST)                       |
| Ollama (binary)   |  latest | Local LLMs (server at `http://localhost:11434`) |
| Git               |  latest | Version control & CI/CD (optional)              |

## Architecture
High-Level Diagram

To Be Added

## Components
- Prompt Template
F-string with the user’s description, platform, tone, content type, and word target. Includes few-shot examples, sets temperature (0.85 default) and requires JSON output (7 items).

- Provider Layer
    - Ollama: /api/chat POST with model name; supports listing local models and pulling new ones
    - OpenAI: chat.completions.create(...) with the selected model (e.g., gpt-4o-mini)

- Post-Processing
    - Parse/repair JSON
    - Enforce Twitter/X 280-char rule
    - Enforce 80–100% word window (trim if over; refine once if under, except on Twitter)

- UI Layer
    - Sidebar inputs (provider toggle, key/model selection, description, platform, tone, content type, word target)
    - Results with expanders, word + char counts, copy buttons, and warnings

## Installation & Setup
Works on Windows/macOS/Linux. We recommend Conda for a clean Python 3.10 environment.

1) Clone the repo
```
git clone https://github.com/malmehayla/social_media_generator.git
cd social_media_generator
```

2) Create & activate Conda environment

Option A — environment.yml (recommended)
```
conda env create -f environment.yml
conda activate smg
```

Option B — manual
```
conda create -n smg python=3.10 -y
conda activate smg
pip install --upgrade pip
pip install -r requirements.txt
```

3) Install & run Ollama (default provider)

Windows: winget install Ollama.Ollama
macOS: brew install ollama
Linux: see https://ollama.com

Pull at least one chat-friendly model (you can also pull inside the app)
```
ollama pull llama3.2:3b
# other good options:
#   qwen2.5:3b-instruct
#   llama3.1:8b
```
Test server:
```
curl http://localhost:11434/api/version
```

4) (Optional) Set your OpenAI API key

Local (session only):

macOS/Linux:
```
export OPENAI_API_KEY="sk-..."
```

Windows PowerShell:
```
setx OPENAI_API_KEY "sk-..."
```
(Restart terminal to load new env var.)

Streamlit Secrets (recommended prod): create .streamlit/secrets.toml (ignored by Git):
```
OPENAI_API_KEY = "sk-..."
```

5) Verify setup

```
streamlit hello
python -c "import streamlit, requests, openai; print('OK')"
ollama run llama3.2:3b "Say OK in one word"
```

## Configuration
- Provider toggle: Default is Ollama. Toggle “Use OpenAI (paid)” to switch.
- Ollama settings: Base URL (http://localhost:11434 by default), local models dropdown (refresh button), Pull model field, timeouts.
- OpenAI settings: API key input, model dropdown (fetched from your key; safe fallbacks like gpt-4o-mini).
- Word target: Auto-sets by Platform × Content Type; editable. The app enforces 80–100% of this target (except on Twitter/X when 280 chars dominates).


### Usage
```
streamlit run app.py
```

Sidebar Inputs
- Provider: Ollama (default) or OpenAI (toggle)
- Product / Campaign Description
- Platform: Twitter/X, Instagram, LinkedIn, TikTok, Facebook
- Content Type: Caption or Post
- Tone: None, fun, professional, inspirational
- Target words per item (auto-adjusts; editable)
- Provider-specific fields (OpenAI model, Ollama pull, timeouts)
- Click Generate.

Output
- 7 items with index, word count, and char count
- Expanders with full text
- Copy button for each item
- Warnings if trimming, truncation, or refinement occurred

Example
- Description: “Eco-friendly running shoes for urban athletes”
- Platform: Instagram
- Content Type: Caption
- Tone: fun

You’ll see 7 short, emoji-rich, CTA-ending captions; some may start with a question; others are bold statements or micro-stories.

## How It Works

1- User Input → platform, content type, tone, target words, description
2- Prompt Template → instructs LLM to return JSON with exactly 7 items
3- Provider Call →
    - Ollama: POST /api/chat
    - OpenAI: chat.completions.create(...)
4- Parsing & Rules →
    - Repair/parse JSON
    - Enforce Twitter/X 280-char limit
    - Enforce 80–100% word target
        - Trim if over
        - Refine pass to expand if under (non-Twitter)
5- Render → expander per item, copy button, warnings

## Platform-specific adjustments
| Platform  | Tweaks                                                             |
| --------- | ------------------------------------------------------------------ |
| Twitter/X | ≤ 280 chars; punchy, high-signal; word target secondary            |
| Instagram | Visual hooks; “Pair with this photo”; playful                      |
| LinkedIn  | Professional/value-driven; compact paragraphs; networking phrasing |
| TikTok    | Fast hooks; playful/trend-aware; “Watch till the end” cues         |
| Facebook  | Conversational, community-oriented; optional single hashtag        |

## Testing & QA
Unit tests (suggested)
Mock Ollama/OpenAI responses and test JSON parsing + enforcement logic.
python script shown in app_test.py
| Case               | Input                  | Expected                                         |
| ------------------ | ---------------------- | ------------------------------------------------ |
| Empty description  | `""`                   | Warning: “Description is too short”              |
| Missing OpenAI key | OpenAI enabled, no key | Error: “Please enter your OpenAI API key.”       |
| Rate limit         | High volume            | Backoff retries; error if still failing          |
| Non-JSON           | Provider drift         | Show error; suggest alternate model              |
| Over target words  | Long item              | Trim to target; show “trimmed to word cap” badge |
| Under 80%          | Very short             | Refine pass (except Twitter) to expand           |

## Performance & Costs
- Ollama: free, local; RAM/CPU usage depends on model size.
- OpenAI: cost depends on model & tokens; one 7-item generation is typically low-cost (prompt + completion).
- Use caching (@st.cache_data) to avoid unnecessary regenerations for identical inputs.

## Security
- Never commit secrets. .streamlit/secrets.toml is ignored by Git.
- Prefer secrets or environment variables over typing keys into the UI when sharing.
- If hosting internally, follow your org’s key management policies.

## Contributing
1- Fork the repo
2- Create a branch: git checkout -b feat/improve-word-refine
3- Commit: git commit -m "feat: better word-range refinement"
4- Push: git push origin feat/improve-word-refine
5- Open a Pull Request with context, screenshots, and testing notes

## License
MIT License — free to use, modify, and distribute with attribution. See LICENSE.

## FAQ
Can I use this fully offline?
- Ollama: Yes, a local model must be pulled in advance. 
- OpenAI: No, requires internet.

Why 7 items? Can I change it?
- It’s a sweet spot for variety. You can change the prompt + JSON validation to any number.

Why are some Twitter/X items short?
- The 280-character limit is enforced over word targets. We’ll trim and warn when needed.

I selected a model, but OpenAI shows 404. Why?
- Your key/org likely doesn’t have access to that model. Pick gpt-4o-mini or another available option.

Where do I change the default word targets?
- In the code, see the RECOMMENDED_WORDS mapping (Platform × Content Type). The UI auto-adjusts.

Does the app store my API key?
- No. The key is used at runtime. For deployed apps, use Streamlit Secrets.

Can I add hashtags or mentions?
- Yes, tweak the prompt to include or expand hashtag rules per platform.

## Future Roadmap
- Multi-language support (input language or auto-detect)
- Visual suggestions or image generation for Instagram/Facebook
- A/B variants per item and export to CSV
- Simple analytics or UTM scaffolding
- Session history and favorites
- Scheduler integrations (Buffer, Hootsuite, X/LinkedIn APIs)
- Plugin system for tone packs and brand style guides

## Quick Start

setup
```
# Create env
conda env create -f environment.yml
conda activate smg
conda init

# Pull a local model (Ollama default)
ollama pull llama3.2:3b
```

verify your setup
```
streamlit hello
python -c "import streamlit, requests, openai; print('OK')"
ollama run llama3.2:3b "Say OK in one word"
```
Note to exit streamlit click ctrl+c

run
```
# Run
streamlit run app.py
```