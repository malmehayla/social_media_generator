# tests/test_json_utils.py
import json
from app import _extract_json

def test_json_parse_direct():
    s = json.dumps({"captions": [{"text": "Hi", "char_count": 2}] * 7})
    data = _extract_json(s)
    assert len(data["captions"]) == 7

# Edge cases
#| Case               | Input                  | Expected                                         |
#| ------------------ | ---------------------- | ------------------------------------------------ |
#| Empty description  | `""`                   | Warning: “Description is too short”              |
#| Missing OpenAI key | OpenAI enabled, no key | Error: “Please enter your OpenAI API key.”       |
#| Rate limit         | High volume            | Backoff retries; error if still failing          |
#| Non-JSON           | Provider drift         | Show error; suggest alternate model              |
#| Over target words  | Long item              | Trim to target; show “trimmed to word cap” badge |
#| Under 80%          | Very short             | Refine pass (except Twitter) to expand           |
