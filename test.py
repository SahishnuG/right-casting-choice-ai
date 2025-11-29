#!/usr/bin/env python3
"""
Run this in your project where src/right_casting_choice_ai is importable.

It will run the crew once and print the raw/parsed output for extract_characters_task.
"""
import os
import re
import json
from pathlib import Path
import sys

# ensure repo src/ is importable (adjust if you run from a different cwd)
sys.path.append(str((Path(__file__).parent / "src").resolve()))

from right_casting_choice_ai.crew import RightCastingChoiceAi
from dotenv import load_dotenv

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ.setdefault("GOOGLE_API_KEY", gemini_key)
serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPERDEV_API_KEY") or ""
if serper_key:
    os.environ["SERPER_API_KEY"] = serper_key
omdb_key = os.getenv("OMDB_API_KEY") or ""
if omdb_key:
    os.environ["OMDB_API_KEY"] = omdb_key

def try_extract_json_from_string(s):
    if s is None:
        return None
    text = str(s).strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate.replace("'", '"'))
            except Exception:
                return None
    return None

def find_task_output(result_dict, task_names=("extract_characters_task", "extract_characters")):
    """
    Look through multiple possible shapes of CrewOutput to find the extract_characters_task output.
    Returns a tuple (raw_value, parsed_json_or_None, location_description)
    """
    # 1) top-level keys
    for name in task_names:
        if isinstance(result_dict, dict) and name in result_dict:
            raw = result_dict[name]
            parsed = try_extract_json_from_string(raw) if isinstance(raw, str) else (raw if isinstance(raw, (list, dict)) else None)
            return raw, parsed, f"top-level key '{name}'"

    # 2) tasks list
    tasks = None
    if isinstance(result_dict, dict):
        tasks = result_dict.get("tasks") or result_dict.get("tasks_output") or result_dict.get("results") and result_dict.get("results").get("tasks")
    if isinstance(tasks, list):
        for t in tasks:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or t.get("task_name") or "").lower()
            if name in [n.lower() for n in task_names] or any(n.lower() in name for n in task_names):
                out = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
                parsed = try_extract_json_from_string(out) if isinstance(out, str) else (out if isinstance(out, (list, dict)) else None)
                return out, parsed, f"tasks list element with name='{t.get('name')}'"

    # 3) results mapping
    if isinstance(result_dict, dict) and isinstance(result_dict.get("results"), dict):
        resmap = result_dict["results"]
        for name in task_names:
            if name in resmap:
                out = resmap[name]
                parsed = try_extract_json_from_string(out) if isinstance(out, str) else (out if isinstance(out, (list, dict)) else None)
                return out, parsed, f"results['{name}']"

    # 4) try to find any JSON-looking list/dict that matches character objects heuristic
    def find_character_list(obj):
        if isinstance(obj, list):
            # heuristics: element is dict with keys like 'role' or 'name_hint' or 'age_range' or 'traits'
            if obj and isinstance(obj[0], dict):
                keys = set(k.lower() for k in obj[0].keys())
                if any(x in keys for x in ("role", "name_hint", "age", "age_range", "traits", "name")):
                    return obj
            for item in obj:
                res = find_character_list(item)
                if res:
                    return res
        elif isinstance(obj, dict):
            for v in obj.values():
                res = find_character_list(v)
                if res:
                    return res
        return None

    found = find_character_list(result_dict)
    if found:
        return found, found, "heuristic search for character-like list"

    return None, None, "not found"


def main():
    # Adjust the plot if you want a different input
    inputs = {
        "plot": "A righteous and fearless police officer takes on corruption and crime to restore justice in his city.",
        "n_similar": 3,
        "usd_to_inr": float(os.getenv("USD_TO_INR_RATE") or 83.0),
        # budgets optional
        "user_budget_inr": None,
        "user_budget_usd": None,
        "industry": "bollywood",
    }

    crew = RightCastingChoiceAi().crew()
    print("Running crew.kickoff(inputs=...)\n(If this hangs or errors, ensure your GEMINI/OMDB/SERPER env vars are set or the crew will fallback.)\n")
    result = crew.kickoff(inputs=inputs)

    # attempt to normalize the whole result to a dict
    result_dict = None
    if hasattr(result, "to_dict"):
        try:
            result_dict = result.to_dict()
        except Exception:
            result_dict = None
    if not result_dict and hasattr(result, "raw"):
        raw = result.raw
        if isinstance(raw, dict):
            result_dict = raw
        elif isinstance(raw, str):
            parsed = try_extract_json_from_string(raw)
            result_dict = parsed or {"raw": raw}
    if not result_dict:
        # try dict(result)
        try:
            result_dict = dict(result)
        except Exception:
            # final fallback: stringify
            s = str(result)
            parsed = try_extract_json_from_string(s)
            result_dict = parsed or {"raw": s}

    print("\n=== FULL NORMALIZED CREW OUTPUT (abridged) ===")
    try:
        print(json.dumps(result_dict, indent=2)[:4000])  # print head only to keep console tidy
    except Exception:
        print(result_dict)

    raw_val, parsed_val, location = find_task_output(result_dict)
    print("\n\n=== extract_characters_task RAW LOCATION ===")
    print(location)
    print("\n--- RAW VALUE ---")
    print(raw_val)
    print("\n--- PARSED (if any) ---")
    print(parsed_val)

    # If parsed_val is a list/dict, pretty-print fully
    if parsed_val and isinstance(parsed_val, (list, dict)):
        print("\n--- PRETTY PARSED JSON (full) ---")
        print(json.dumps(parsed_val, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
