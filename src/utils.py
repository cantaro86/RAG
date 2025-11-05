import json
import re


def extract_last_json(raw_text: str):
    """
    Extract the last valid JSON object from the model output.
    If none found or invalid, return {"score": "no"}.
    """
    # Match any {...} including across line breaks
    matches = re.findall(r"\{[^{}]+\}", raw_text, re.DOTALL)
    if not matches:
        print("No JSON object found in output.")
        return {"score": "no"}

    last = matches[-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        # Clean potential trailing commas or artifacts
        cleaned = re.sub(r",\s*}", "}", last.strip())
        try:
            return json.loads(cleaned)
        except Exception:
            print("Failed to decode last JSON object.")
            return {"score": "no"}


def extract_answer_text(raw_text: str):
    """
    Extracts the text following the 'Answer:' section in model output.
    If no explicit 'Answer:' found, returns the full text.
    """
    # Try to find 'Answer:' ignoring case
    match = re.search(r"(?i)answer\s*:\s*(.*)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # If no explicit "Answer:" header, return full output
        return raw_text.strip()
