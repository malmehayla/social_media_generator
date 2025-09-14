# app.py
# Author: Mohamed Almehayla
# Social Media Generator – Default: Ollama (free/local), Toggle: OpenAI (paid)
# Platforms: Twitter/X, Instagram, LinkedIn, TikTok, Facebook
# Content type: Caption/Post; adjustable target words with 80–100% enforcement (refine pass)
# OpenAI: dropdown of available models (pulled from your key when present; excludes gpt-5)
# Ollama: list installed models, refresh list, and pull models from the UI

import os
import json
import re
import math
from typing import List, Dict, Tuple, Optional

import requests
from requests.exceptions import RequestException

import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError  # <-- precise exception
from openai import OpenAI
from openai import APIError, AuthenticationError, RateLimitError

# -------------------------------
# Config / Constants
# -------------------------------
PLATFORMS = ["Twitter/X", "Instagram", "LinkedIn", "TikTok", "Facebook"]
CONTENT_TYPES = ["Caption", "Post"]
TONES = ["None", "fun", "professional", "inspirational"]
TWITTER_MAX = 280

RECOMMENDED_WORDS = {
    "Caption": {"Twitter/X": 18, "Instagram": 30, "LinkedIn": 35, "TikTok": 18, "Facebook": 30},
    "Post": {"Twitter/X": 40, "Instagram": 400, "LinkedIn": 450, "TikTok": 120, "Facebook": 450},
}

FALLBACK_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o3-mini"]
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


# -------------------------------
# Secrets & Key Resolution (ROCK-SOLID)
# -------------------------------
def _mask_key(k: Optional[str]) -> str:
    if not k:
        return "(none)"
    k = str(k)
    if len(k) <= 10:
        return "***"
    return f"{k[:4]}***{k[-4:]}"


def _read_from_secrets() -> Tuple[str, str]:
    """
    Safely read from Streamlit secrets in multiple shapes without ever crashing
    if .streamlit/secrets.toml is missing. Returns (key, source_str).
    Supported shapes:
      openai_api_key = "sk-..."
      OPENAI_API_KEY = "sk-..."
      [openai] api_key = "sk-..."
      [openai] OPENAI_API_KEY = "sk-..."
    """
    # Any attempt to access st.secrets keys will trigger parsing; guard each access.
    try:
        s = st.secrets  # obtaining the proxy is fine; parsing happens on access
    except Exception:
        return "", "secrets:unavailable"

    # top-level keys
    for name in ("openai_api_key", "OPENAI_API_KEY"):
        try:
            v = s.get(name)  # this may raise StreamlitSecretNotFoundError if secrets file missing
        except StreamlitSecretNotFoundError:
            return "", "secrets:not_found"
        except Exception:
            return "", "secrets:unavailable"
        if isinstance(v, str) and v.strip():
            return v.strip(), f"secrets:{name}"

    # nested sections
    for sect in ("openai", "OpenAI", "OPENAI"):
        try:
            sec = s[sect]  # bracket access can also raise
        except StreamlitSecretNotFoundError:
            return "", "secrets:not_found"
        except Exception:
            sec = None
        if isinstance(sec, dict):
            for name in ("api_key", "API_KEY", "openai_api_key", "OPENAI_API_KEY"):
                v = sec.get(name)
                if isinstance(v, str) and v.strip():
                    return v.strip(), f"secrets:{sect}.{name}"

    # secrets exists but no matching keys
    return "", "secrets:not_found"


def _read_from_env() -> Tuple[str, str]:
    for name in ("OPENAI_API_KEY", "openai_api_key"):
        v = os.getenv(name, "").strip()
        if v:
            return v, f"env:{name}"
    return "", "env:not_found"


def resolve_openai_key() -> Tuple[str, str]:
    """
    Deterministic resolution order:
      1) Streamlit secrets (multiple shapes)
      2) Environment variables
      3) Empty string if not found
    Returns (key, source_string)
    """
    k, src = _read_from_secrets()
    if k:
        return k, src
    k, src = _read_from_env()
    if k:
        return k, src
    return "", "none"


# -------------------------------
# Prompt Builders
# -------------------------------
def get_prompt(description: str, platform: str, tone: Optional[str],
               content_type: str, word_target: int, fenced: bool = False,
               min_ratio: float = 0.8) -> str:
    tone_text = tone if tone and tone.lower() != "none" else "neutral"
    min_words = max(1, math.floor(min_ratio * word_target))
    few_shots = """
Few-shot examples (style and formatting hints):

Example 1
Input:
  Description: "Eco-friendly running shoes for urban athletes"
  Platform: Instagram
  Tone: fun
  Content Type: Caption
  Target Words: ~30
Output (one caption example only):
  "City runs, but make it green! 🌿👟 Lightweight, comfy, and kind to the planet. Ready to sprint the skyline? Tag your run buddy & shop now! #EcoRun #UrbanAthlete"

Example 2
Input:
  Description: "AI-powered CRM that saves time for sales teams"
  Platform: LinkedIn
  Tone: professional
  Content Type: Post
  Target Words: ~450
Output (one item example only):
  "Prioritize relationships, not admin. 🤝⚙️ Our AI-powered CRM automates the busywork so your team can focus on revenue. See how it scales your pipeline—request a demo today."
""".strip()

    base = f"""
You are a creative social media expert and copywriter.
Generate EXACTLY 7 diverse, high-quality items for the product/campaign below.

Product/Campaign: {description}
Platform: {platform}
Tone: {tone_text}
Content Type: {content_type}.

Platform adjustments:
- Twitter/X: under 280 characters; short, punchy, high-signal copy (ALWAYS prioritize 280-char limit over word target).
- Instagram: visual hooks, descriptive cues like "Pair with this stunning photo", playful vibes.
- LinkedIn: professional, value-driven, concise sentences; networking/insight phrasing.
- TikTok: hook fast, playful and trend-aware; "Watch till the end", "Follow for more".
- Facebook: conversational and community-oriented; optional single hashtag; avoid overusing emojis.

Length requirements:
- Aim for between {min_words} and {word_target} words per {content_type.lower()} (inclusive).
- HARD WORD CAP: Do not exceed {word_target} words.
- Lower bound: Do NOT go below {min_words} words, UNLESS on Twitter/X where 280 chars make this impossible.

Structure requirements:
- If Content Type is "Caption": keep it short and hooky (1–3 sentences max).
- If Content Type is "Post": use a strong headline, a short intro, 2–3 scannable paragraphs, optional 3–5 punchy bullet points, and a closing CTA line.

Each item MUST:
- Include 2–3 relevant emojis (no more, no less).
- End with a strong CTA (e.g., "Shop now!", "Tag a friend!", "Request a demo!", "Learn more today!").
- Vary the structure across the set (some with a question, some with a bold statement, some with a story hook).
- Avoid hashtags unless the platform naturally benefits (for MVP, keep hashtags minimal or omit).

{few_shots}

Output format requirements (MANDATORY):
- Respond ONLY with JSON using this schema:

{{
  "captions": [
    {{"text": "CAPTION_1", "char_count": 0}},
    ... (total 7 objects)
  ]
}}

- "text" is the caption/post string (use paragraphs and bullets if "Post").
- "char_count" is the character count of the item (len of the text).
""".strip()

    if fenced:
        base += "\nReturn ONLY the JSON inside triple backticks like:\n```json\n{ ... }\n```\nNo other text."
    else:
        base += "\nDo not include any backticks, code fences, or extra commentary. Return exactly 7 items."
    return base


def get_refine_prompt(items: List[str], description: str, platform: str, tone: Optional[str],
                      content_type: str, min_words: int, max_words: int, fenced: bool = False) -> str:
    tone_text = tone if tone and tone.lower() != "none" else "neutral"
    joined = "\n".join([f"{i+1}. {it}" for i, it in enumerate(items)])
    base = f"""
Refine the following {len(items)} items for:
- Product/Campaign: {description}
- Platform: {platform}
- Tone: {tone_text}
- Content Type: {content_type}

GOAL:
- For each item, adjust wording so the word count is between {min_words} and {max_words} words (inclusive).
- Twitter/X EXCEPTION: If {platform} == Twitter/X, NEVER exceed 280 characters. It's OK if the item falls below {min_words} due to that limit.
- Keep 2–3 emojis per item and end with a clear CTA.
- Preserve the original intent; improve clarity and flow. Vary openings where possible.

ITEMS TO REFINE:
{joined}

Return ONLY JSON:
{{
  "captions": [
    {{"text": "REFINED_1", "char_count": 0}},
    ...
  ]
}}
""".strip()
    if fenced:
        base += "\nWrap the JSON ONLY in triple backticks like:\n```json\n{ ... }\n```"
    return base


# -------------------------------
# JSON Repair / Parsing Helpers
# -------------------------------
def _strip_code_fences(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    return s

def _normalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

def _remove_json_comments(s: str) -> str:
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([\]\}])", r"\1", s)

def _extract_largest_brace_block(s: str) -> str:
    start, end = s.find("{"), s.rfind("}")
    return s[start:end+1] if start != -1 and end != -1 and end > start else s

def _repair_and_load_json(raw: str) -> dict:
    s = _strip_code_fences(raw.strip())
    s = _normalize_quotes(s)
    s = _remove_json_comments(s)
    s = _extract_largest_brace_block(s)
    s = _remove_trailing_commas(s)
    return json.loads(s)

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return _repair_and_load_json(m.group(1))
        except Exception:
            pass
    return _repair_and_load_json(text)

def _normalize_captions(parsed: dict) -> List[Dict[str, str]]:
    caps = parsed.get("captions", [])
    if not isinstance(caps, list):
        raise ValueError("JSON does not contain a list under 'captions'.")
    out: List[Dict[str, str]] = []
    for item in caps:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip() or str(item.get("caption", "")).strip()
        else:
            text = str(item).strip()
        if not text:
            continue
        out.append({"text": text, "char_count": len(text)})
    if len(out) < 7:
        raise ValueError(f"Model returned only {len(out)} items; expected 7.")
    return out[:7]

def word_count(s: str) -> int:
    return len(re.findall(r"\S+", s))

def _enforce_platform_rules(items: List[Dict[str, str]], platform: str) -> Tuple[List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    processed: List[Dict[str, str]] = []
    for cap in items:
        text = cap.get("text", "").strip()
        meta = dict(cap)
        if platform == "Twitter/X" and len(text) > TWITTER_MAX:
            truncated = text[:TWITTER_MAX]
            meta["text"] = truncated
            meta["char_count"] = len(truncated)
            meta["truncated"] = True
            warnings.append("One or more Twitter/X items exceeded 280 chars and were truncated.")
        processed.append(meta if "text" in meta else {"text": text, "char_count": len(text)})
    return processed, list(set(warnings))

def _enforce_word_bounds(items: List[Dict[str, str]], min_words: int, max_words: int, platform: str
                         ) -> Tuple[List[Dict[str, str]], List[str], List[int]]:
    warnings: List[str] = []
    too_short_idxs: List[int] = []
    processed: List[Dict[str, str]] = []
    for idx, cap in enumerate(items):
        text = cap.get("text", "").strip()
        wc = word_count(text)
        meta = dict(cap)
        if wc > max_words:
            words = re.findall(r"\S+", text)
            trimmed = " ".join(words[:max_words]).rstrip(",;:-")
            if not re.search(r"[.!?]$", trimmed):
                trimmed += "…"
            meta["text"] = trimmed
            meta["char_count"] = len(trimmed)
            meta["trimmed_words"] = True
        elif wc < min_words and platform != "Twitter/X":
            meta["too_short"] = True
            too_short_idxs.append(idx)
        processed.append(meta if "text" in meta else {"text": text, "char_count": len(text)})
    if any(m.get("trimmed_words") for m in processed):
        warnings.append("One or more items exceeded the target word cap and were trimmed.")
    if too_short_idxs:
        warnings.append("Some items were below 80% of the target and were expanded.")
    return processed, list(set(warnings)), too_short_idxs


# -------------------------------
# Providers & Utilities
# -------------------------------
def check_ollama_health(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise RuntimeError(
            f"Ollama server is not reachable at {base_url}. "
            f"Make sure Ollama is running and the URL is correct. Details: {e}"
        )

def list_ollama_local_models(base_url: str) -> List[str]:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        names = [m.get("name") for m in models if m.get("name")]
        return sorted(set(names))
    except Exception:
        return []

def pull_ollama_model(base_url: str, model: str, timeout_sec: int = 1800) -> str:
    url = f"{base_url.rstrip('/')}/api/pull"
    payload = {"model": model, "stream": True}
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_sec) as r:
            r.raise_for_status()
            status_lines = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line.decode("utf-8"))
                    status = evt.get("status") or evt.get("digest") or ""
                    if "error" in evt:
                        status_lines.append(f"Error: {evt['error']}")
                    elif status:
                        status_lines.append(str(status))
                except Exception:
                    pass
            return "\n".join(status_lines) if status_lines else "Pull completed (no status)."
    except RequestException as e:
        try:
            r = requests.post(f"{base_url.rstrip('/')}/api/pull", json={"model": model, "stream": False}, timeout=timeout_sec)
            r.raise_for_status()
            return "Pull request accepted (non-streaming)."
        except Exception as e2:
            raise RuntimeError(f"Pull failed: {e2}") from e

def is_probably_chat_model(model_id: str) -> bool:
    nid = model_id.lower()
    if any(tok in nid for tok in ["embedding", "whisper", "tts", "audio", "omni-moderation"]):
        return False
    return any(tok in nid for tok in ["gpt", "o3", "4o", "4.1", "3.5"])

def fetch_openai_chat_models(api_key: str) -> List[str]:
    try:
        client = OpenAI(api_key=api_key, timeout=30)
        res = client.models.list()
        ids = [m.id for m in getattr(res, "data", []) if getattr(m, "id", None)]
        chat_ids = [mid for mid in ids if is_probably_chat_model(mid)]
        chat_ids = [m for m in chat_ids if not m.lower().startswith("gpt-5")]
        favorites = [m for m in FALLBACK_OPENAI_MODELS if m in chat_ids]
        others = sorted(set(chat_ids) - set(favorites))
        ordered = favorites + others
        return ordered or [m for m in FALLBACK_OPENAI_MODELS if not m.lower().startswith("gpt-5")]
    except Exception:
        return [m for m in FALLBACK_OPENAI_MODELS if not m.lower().startswith("gpt-5")]

def _openai_chat(api_key: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    client = OpenAI(api_key=api_key, timeout=60)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message.content

def generate_captions_openai(api_key: str, description: str, platform: str, tone: str,
                             content_type: str, word_target: int, model: str
                             ) -> Tuple[List[Dict[str, str]], List[str]]:
    prompt = get_prompt(description, platform, tone, content_type, word_target, fenced=False, min_ratio=0.8)
    est_tokens = int(min(max(word_target * 7 * 2, 1200), 7000))
    content = _openai_chat(api_key, model, prompt, temperature=0.85, max_tokens=est_tokens)
    data = _extract_json(content)
    normalized = _normalize_captions(data)
    stage1, warns_p = _enforce_platform_rules(normalized, platform)
    min_words = max(1, math.floor(0.8 * word_target))
    stage2, warns_w1, too_short = _enforce_word_bounds(stage1, min_words, word_target, platform)
    if too_short and platform != "Twitter/X":
        items_to_refine = [stage2[i]["text"] for i in range(len(stage2))]
        refine_prompt = get_refine_prompt(items_to_refine, description, platform, tone, content_type,
                                          min_words=min_words, max_words=word_target, fenced=False)
        refine_tokens = int(min(max(word_target * 7 * 2, 1200), 7000))
        refined_content = _openai_chat(api_key, model, refine_prompt, temperature=0.6, max_tokens=refine_tokens)
        refined_json = _extract_json(refined_content)
        refined_items = _normalize_captions(refined_json)
        r1, warns_p2 = _enforce_platform_rules(refined_items, platform)
        r2, warns_w2, _ = _enforce_word_bounds(r1, min_words, word_target, platform)
        return r2, list(set(warns_p + warns_w1 + warns_p2 + warns_w2))
    return stage2, list(set(warns_p + warns_w1))

def _ollama_chat(base_url: str, model: str, prompt: str, temperature: float,
                 connect_timeout: int, read_timeout: int, fenced: bool) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_ctx": 8192},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def generate_captions_ollama(description: str, platform: str, tone: str, content_type: str, word_target: int,
                             base_url: str = "http://localhost:11434", model: str = DEFAULT_OLLAMA_MODEL,
                             connect_timeout: int = 10, read_timeout: int = 300
                             ) -> Tuple[List[Dict[str, str]], List[str]]:
    check_ollama_health(base_url)
    prompt = get_prompt(description, platform, tone, content_type, word_target, fenced=True, min_ratio=0.8)
    content = _ollama_chat(base_url, model, prompt, temperature=0.85,
                           connect_timeout=connect_timeout, read_timeout=read_timeout, fenced=True)
    data = _extract_json(content)
    normalized = _normalize_captions(data)
    stage1, warns_p = _enforce_platform_rules(normalized, platform)
    min_words = max(1, math.floor(0.8 * word_target))
    stage2, warns_w1, too_short = _enforce_word_bounds(stage1, min_words, word_target, platform)
    if too_short and platform != "Twitter/X":
        items_to_refine = [stage2[i]["text"] for i in range(len(stage2))]
        refine_prompt = get_refine_prompt(items_to_refine, description, platform, tone, content_type,
                                          min_words=min_words, max_words=word_target, fenced=True)
        refined_content = _ollama_chat(base_url, model, refine_prompt, temperature=0.6,
                                       connect_timeout=connect_timeout, read_timeout=read_timeout, fenced=True)
        refined_json = _extract_json(refined_content)
        refined_items = _normalize_captions(refined_json)
        r1, warns_p2 = _enforce_platform_rules(refined_items, platform)
        r2, warns_w2, _ = _enforce_word_bounds(r1, min_words, word_target, platform)
        return r2, list(set(warns_p + warns_w1 + warns_p2 + warns_w2))
    return stage2, list(set(warns_p + warns_w1))


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Social Media Generator", page_icon="📝", layout="wide")
st.title("📝 Social Media Generator")
st.caption("By: **Mohamed Almehayla**")

if "busy" not in st.session_state:
    st.session_state["busy"] = False

def default_words(platform: str, content_type: str) -> int:
    return RECOMMENDED_WORDS.get(content_type, {}).get(platform, 30)

with st.sidebar:
    st.header("Inputs")

    # ---- Always resolve secrets/env on EVERY rerun (no stale state) ----
    auto_key, auto_source = resolve_openai_key()

    use_openai = st.toggle(
        "Use OpenAI (paid)",
        value=False,
        help="Enable to use OpenAI GPT models. When OFF, the app uses your local Ollama model.",
        key="use_openai",
    )

    # User override never touches st.secrets; it avoids crashes and staleness.
    user_key = st.text_input(
        "OpenAI API Key (optional override)",
        type="password",
        help="If left blank, the app uses your Secrets/env key automatically.",
        key="openai_key_input",
    )

    # Effective key selection order: user override > auto discovered
    effective_openai_key = (user_key.strip() if user_key and user_key.strip() else auto_key).strip()

    # Keep environment in sync for libraries that read from env
    if effective_openai_key:
        os.environ["OPENAI_API_KEY"] = effective_openai_key

    # Diagnostics (masked)
    with st.expander("🔎 Key diagnostics (safe to share)"):
        st.write({
            "auto_source": auto_source,
            "auto_key_masked": _mask_key(auto_key),
            "user_override_provided": bool(user_key.strip()),
            "effective_key_masked": _mask_key(effective_openai_key),
            "env_OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
        })
        st.caption("If auto_source is 'none' or 'secrets:not_found', add `.streamlit/secrets.toml` or set the `OPENAI_API_KEY` env var.")

    if use_openai and not effective_openai_key:
        st.info(
            "No OpenAI API key detected. Add it via `.streamlit/secrets.toml` "
            "(e.g., `openai_api_key = \"sk-...\"`), set the `OPENAI_API_KEY` environment variable, "
            "or paste it above."
        )

    # Common inputs
    description = st.text_area(
        "Product / Campaign Description",
        placeholder="e.g., Eco-friendly running shoes for urban athletes",
        height=140,
        key="desc",
    )
    platform = st.selectbox("Platform", PLATFORMS, index=0, key="platform")
    content_type = st.selectbox("Content type", CONTENT_TYPES, index=0, key="content_type")
    tone = st.selectbox("Tone (optional)", TONES, index=0, key="tone")

    rec_target = default_words(platform, content_type)
    if "wt_context" not in st.session_state:
        st.session_state["wt_context"] = (platform, content_type)
    if "word_target" not in st.session_state:
        st.session_state["word_target"] = rec_target
    if (platform, content_type) != st.session_state["wt_context"]:
        st.session_state["wt_context"] = (platform, content_type)
        st.session_state["word_target"] = rec_target

    st.number_input("Target words per item", min_value=5, max_value=2000, step=5, key="word_target")
    st.caption(f"Recommended for {platform} ({content_type.lower()}): ~{rec_target} words. (App enforces 80–100% range.)")

    if use_openai:
        openai_models = fetch_openai_chat_models(effective_openai_key) if effective_openai_key else [
            m for m in FALLBACK_OPENAI_MODELS if not m.lower().startswith("gpt-5")
        ]
        preferred = "gpt-4o-mini" if "gpt-4o-mini" in openai_models else openai_models[0]
        openai_model = st.selectbox(
            "OpenAI model",
            openai_models,
            index=openai_models.index(preferred) if preferred in openai_models else 0,
            key="openai_model_select",
            help="Choose the OpenAI model used for generation."
        )
        ollama_base = None
        ollama_model = None
        read_timeout = None
        connect_timeout = None
    else:
        ollama_base = st.text_input("Ollama base URL", value="http://localhost:11434", key="ollama_base")

        def list_ollama_local_models_safe(base: str) -> List[str]:
            try:
                return list_ollama_local_models(base)
            except Exception:
                return []

        refresh_clicked = st.button("🔄 Refresh local models", help="Reload installed models from Ollama")
        if refresh_clicked or "ollama_local_models" not in st.session_state:
            st.session_state["ollama_local_models"] = list_ollama_local_models_safe(ollama_base)

        local_models = st.session_state.get("ollama_local_models", [])
        if not local_models:
            st.info("No local models detected. Pull one below to get started.")
        if DEFAULT_OLLAMA_MODEL not in local_models:
            local_models = [DEFAULT_OLLAMA_MODEL] + local_models

        ollama_model = st.selectbox(
            "Local Ollama model",
            local_models,
            index=local_models.index(DEFAULT_OLLAMA_MODEL) if DEFAULT_OLLAMA_MODEL in local_models else 0,
            key="ollama_model_select"
        )

        st.markdown("**Pull a model from Ollama registry** (e.g., `llama3.2:3b`, `qwen2.5:3b-instruct`)")
        pull_name = st.text_input("Model name to pull", value=DEFAULT_OLLAMA_MODEL, key="ollama_pull_name")
        pull_btn = st.button("⬇️ Pull model")
        if pull_btn:
            try:
                with st.spinner(f"Pulling `{pull_name}` from Ollama registry..."):
                    status = pull_ollama_model(ollama_base, pull_name)
                st.success(f"Pull completed for `{pull_name}`")
                with st.expander("Pull output", expanded=False):
                    st.text(status or "Done.")
                st.session_state["ollama_local_models"] = list_ollama_local_models_safe(ollama_base)
            except Exception as e:
                st.error(str(e))

        read_timeout = st.number_input("Read timeout (seconds)", min_value=30, max_value=600, value=300, step=30, key="ollama_read_to")
        connect_timeout = st.number_input("Connect timeout (seconds)", min_value=2, max_value=60, value=10, step=2, key="ollama_connect_to")
        st.caption("Tip: pull the model first:  e.g., `ollama pull llama3.2:3b`")
        openai_model = None

    generate_clicked = st.button("Generate", disabled=st.session_state["busy"])

st.markdown("> Tips: Twitter/X is char-limited (we'll truncate if needed). The app enforces an 80–100% word range (except the X 280-char limit).")


# Copy button helper
def render_copy_button(text: str, key: str):
    escaped = json.dumps(text)
    html = f"""
    <div style="display:flex;align-items:center;gap:8px">
      <button id="copy-{key}" style="padding:6px 10px;border-radius:6px;border:1px solid #ccc;cursor:pointer;">Copy</button>
      <span id="status-{key}" style="font-size:0.9rem;color:#4CAF50;"></span>
    </div>
    <script>
      const btn = document.getElementById("copy-{key}");
      const status = document.getElementById("status-{key}");
      if (btn) {{
        btn.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText({escaped});
            status.textContent = "Copied!";
            setTimeout(() => {{ status.textContent = ""; }}, 1500);
          }} catch (e) {{
            status.textContent = "Copy failed";
            status.style.color = "#E63946";
            setTimeout(() => {{ status.textContent = ""; status.style.color="#4CAF50"; }}, 1500);
          }}
        }});
      }}
    </script>
    """
    components.html(html, height=48)


# -------------------------------
# Cache
# -------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def cached_generate(description: str, platform: str, tone: str,
                    content_type: str, word_target: int,
                    provider: str,
                    api_key: Optional[str],
                    openai_model: Optional[str],
                    ollama_base: Optional[str],
                    ollama_model: Optional[str],
                    connect_timeout: Optional[int],
                    read_timeout: Optional[int]):
    if provider == "ollama":
        return generate_captions_ollama(
            description, platform, tone, content_type, word_target,
            base_url=ollama_base or "http://localhost:11434",
            model=ollama_model or DEFAULT_OLLAMA_MODEL,
            connect_timeout=connect_timeout or 10,
            read_timeout=read_timeout or 300,
        )
    if not api_key:
        raise RuntimeError("Missing OpenAI API key")
    return generate_captions_openai(
        api_key, description, platform, tone, content_type, word_target,
        model=openai_model or "gpt-4o-mini"
    )


# -------------------------------
# Main action
# -------------------------------
if generate_clicked:
    if not description or len(description.strip()) < 10:
        st.warning("Description is too short. Please write at least 10 characters.")
    else:
        provider = "openai" if st.session_state["use_openai"] else "ollama"

        # KEY CHECK RIGHT BEFORE CALL (final gate)
        if provider == "openai" and not effective_openai_key:
            st.error("Please set your OpenAI API key (Secrets/env/UI).")
        else:
            if st.session_state["busy"]:
                st.stop()
            st.session_state["busy"] = True
            try:
                with st.spinner("Generating items..."):
                    caps, warns = cached_generate(
                        description.strip(), platform, tone,
                        content_type, int(st.session_state["word_target"]),
                        provider,
                        effective_openai_key if provider == "openai" else None,
                        openai_model,
                        ollama_base, ollama_model,
                        connect_timeout, read_timeout
                    )
            except RuntimeError as e:
                if (provider == "ollama") and effective_openai_key:
                    st.warning(f"{e}  • Falling back to OpenAI…")
                    caps, warns = cached_generate(
                        description.strip(), platform, tone,
                        content_type, int(st.session_state["word_target"]),
                        "openai", effective_openai_key, "gpt-4o-mini",
                        None, None, None, None
                    )
                else:
                    st.error(str(e))
                    st.session_state["busy"] = False
                    st.stop()
            except AuthenticationError:
                st.error("Authentication failed. Please check your OpenAI API key.")
                st.session_state["busy"] = False
                st.stop()
            except RateLimitError:
                st.error("Rate limit exceeded after multiple retries. Try again shortly.")
                st.session_state["busy"] = False
                st.stop()
            except APIError as e:
                st.error(f"OpenAI API error: {e}")
                st.session_state["busy"] = False
                st.stop()
            except json.JSONDecodeError:
                st.error("The model returned invalid JSON. Try again.")
                st.session_state["busy"] = False
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state["busy"] = False
                st.stop()

            if 'caps' in locals():
                if warns:
                    for w in warns:
                        st.warning(w)

                st.subheader("Your Output")
                if not caps:
                    st.info("No items returned. Try again or simplify your description.")
                else:
                    for i, c in enumerate(caps, start=1):
                        text = c.get("text", "").strip()
                        char_count = c.get("char_count", len(text))
                        wcount = word_count(text)
                        badges = []
                        if c.get("trimmed_words"):
                            badges.append("trimmed to word cap")
                        if c.get("truncated"):
                            badges.append("truncated to 280 chars")
                        if c.get("too_short"):
                            badges.append("below 80% (after best effort)")
                        badge_str = f" • {' • '.join(badges)}" if badges else ""
                        label = f"{i}. {wcount} words • {char_count} chars{badge_str}"
                        with st.expander(label, expanded=False):
                            st.markdown(text)
                            render_copy_button(text, key=f"{i}")

                st.success("Done! Feel free to tweak inputs and regenerate.")
            st.session_state["busy"] = False

with st.sidebar:
    st.caption(
        "Default provider: Ollama (local). Toggle 'Use OpenAI (paid)' to switch. "
        "App enforces an 80–100% word range (except X’s 280-char limit). "
        "Use Refresh to update local model list; Pull to install new models. "
        "Only OpenAI will work for streamlit deployed app. "
        "Built by **Mohamed Almehayla**"
    )
