# app.py — robust parsing + fees + USD->INR boxoffice parsing + improved UI
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- env & import path ----
load_dotenv()
sys.path.append(str((Path(__file__).parent / "src").resolve()))
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if gemini_key:
    os.environ.setdefault("GEMINI_API_KEY", gemini_key)
    os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

# project imports (keep as in your repo)
from right_casting_choice_ai.crew import RightCastingChoiceAi

# ---- helpers ----
def try_extract_json_from_string(s: Optional[str]) -> Optional[Any]:
    """
    Robustly extract JSON payload from messy LLM output strings.
    Strategy:
    1) Remove all code fences (```json ... ``` or ``` ... ``` anywhere in text).
    2) Try full-string json.loads.
    3) Try last fenced JSON block if present (```json ... ```), then parse.
    4) Scan for JSON arrays across the text (prefer last successful parse), then objects.
    5) Retry with single-quote to double-quote replacement for candidates.
    """
    if s is None:
        return None
    text = str(s)
    # If there are code-fenced blocks, try the last fenced block first
    fenced_blocks = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if not fenced_blocks:
        fenced_blocks = re.findall(r"```\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced_blocks:
        last_block = fenced_blocks[-1].strip()
        for candidate in (last_block, last_block.replace("'", '"')):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    # Remove all fences globally and try whole string
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    for candidate in (cleaned, cleaned.replace("'", '"')):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Find all JSON arrays, prefer the last successful parse
    arrays = list(re.finditer(r"\[[\s\S]*?\]", cleaned))
    for m in reversed(arrays):
        cand = m.group(0)
        for candidate in (cand, cand.replace("'", '"')):
            try:
                parsed = json.loads(candidate)
                return parsed
            except Exception:
                pass
    # Then find JSON objects
    objects = list(re.finditer(r"\{[\s\S]*?\}", cleaned))
    for m in reversed(objects):
        cand = m.group(0)
        for candidate in (cand, cand.replace("'", '"')):
            try:
                parsed = json.loads(candidate)
                return parsed
            except Exception:
                pass
    return None

# Intentionally no plot-based name extraction; rely solely on the extractor agent

def safe_crew_kickoff(inputs: Dict[str, Any], max_retries: int = 1) -> Dict[str, Any]:
    """Run crew with up to one retry on Gemini rate limits, respecting suggested delay."""
    try:
        crew = RightCastingChoiceAi().crew()
        result = None
        last_error = None
        attempts = max(1, int(max_retries) + 1)
        for attempt in range(attempts):
            try:
                result = crew.kickoff(inputs=inputs)
                last_error = None
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                is_rate_limit = ("RESOURCE_EXHAUSTED" in msg or "RateLimit" in msg or "quota" in msg.lower())
                if not is_rate_limit or attempt >= attempts - 1:
                    break
                # Parse suggested retry delay from Gemini error JSON if present
                retry_seconds = None
                try:
                    jstart = msg.find('{')
                    if jstart != -1:
                        data = json.loads(msg[jstart:])
                        details = (data.get("error") or {}).get("details") or []
                        for d in details:
                            if d.get("@type", "").endswith("RetryInfo"):
                                rd = d.get("retryDelay")
                                if isinstance(rd, str) and rd.endswith("s"):
                                    retry_seconds = float(rd[:-1])
                                elif isinstance(rd, (int, float)):
                                    retry_seconds = float(rd)
                                break
                except Exception:
                    retry_seconds = None
                time.sleep(retry_seconds if retry_seconds else 30.0)
        # prefer to_dict
        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass
        # raw attr
        if hasattr(result, "raw"):
            raw = result.raw
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, list):
                return {"raw": raw}
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                return {"raw": parsed} if parsed is not None else {"raw": raw}
        # dict(result)
        try:
            d = dict(result)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        s = str(result)
        parsed = try_extract_json_from_string(s)
        if parsed is not None:
            return {"raw": parsed}
        if last_error is not None:
            return {"error": f"Crew kickoff failed: {last_error}"}
        return {"raw": str(result)}
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}

def pick_first(obj: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return v
    return default

def parse_money_str_to_int(s: Optional[str], usd_to_inr: float = 83.0) -> Optional[int]:
    """
    Parse strings like "$5,102,129" or "₹12,345,678" or "12,345,678" -> integer INR.
    If string contains $ assume USD and convert to INR using usd_to_inr.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.upper() == "N/A":
        return None
    # detect USD symbol
    is_usd = bool(re.search(r"^\s*\$", s) or re.search(r"\$[0-9,]", s))
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    try:
        val = int(digits)
        if is_usd:
            return int(val * usd_to_inr)
        # if ₹ or no currency assume INR
        return int(val)
    except Exception:
        return None

def normalize_movie_boxoffice(m: Dict[str, Any], usd_to_inr: float) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (box_inr, budget_inr) trying:
      - BoxOfficeINR / BudgetINR fields (numeric)
      - BoxOffice / Budget strings parsed (USD->INR or INR)
      - _raw.BoxOffice if present
    """
    # box
    box = None
    for key in ("BoxOfficeINR", "box_office_inr", "box_inr"):
        v = m.get(key)
        if isinstance(v, (int, float)):
            box = int(v)
            break
    if box is None:
        candidates = [m.get("BoxOffice"), m.get("box_office"), pick_first(m.get("_raw") or {}, ["BoxOffice", "box_office"], None)]
        for s in candidates:
            if s:
                parsed = parse_money_str_to_int(s, usd_to_inr)
                if parsed:
                    box = parsed
                    break
    # budget
    budget = None
    for key in ("BudgetINR", "budget_inr", "budget"):
        v = m.get(key)
        if isinstance(v, (int, float)):
            budget = int(v)
            break
    if budget is None:
        candidates = [m.get("Budget"), pick_first(m.get("_raw") or {}, ["Budget", "budget"], None)]
        for s in candidates:
            if s:
                parsed = parse_money_str_to_int(s, usd_to_inr)
                if parsed:
                    budget = parsed
                    break
    return box, budget

def extract_roles_and_movies_from_crew(crew_out: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    roles: List[Dict[str, Any]] = []
    movies: List[Dict[str, Any]] = []

    if not isinstance(crew_out, dict):
        return roles, movies

    raw = crew_out.get("raw")
    # common shapes:
    if isinstance(raw, list):
        # raw is likely the top-level role list (each item has role/candidates/movies)
        if raw and isinstance(raw[0], dict) and ("role" in raw[0] or "candidates" in raw[0] or "name_hint" in raw[0]):
            roles = raw
        else:
            # try to find role-like entries and movie-like entries
            for v in raw:
                if isinstance(v, dict):
                    if "role" in v or "candidates" in v or "name_hint" in v:
                        roles.append(v)
                    if "imdbID" in v or "Title" in v or "Poster" in v:
                        movies.append(v)
    elif isinstance(raw, dict):
        # raw is dict — it may contain movies/candidates keys
        if isinstance(raw.get("movies"), list):
            movies = raw["movies"]
        if isinstance(raw.get("candidates"), list):
            roles = [{"role": raw.get("role") or raw.get("name_hint") or "Lead", "candidates": raw["candidates"], "movies": raw.get("movies", [])}]
    elif isinstance(raw, str):
        parsed = try_extract_json_from_string(raw)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict) and ("role" in parsed[0] or "candidates" in parsed[0]):
                roles = parsed
            else:
                for v in parsed:
                    if isinstance(v, dict):
                        if "role" in v or "candidates" in v:
                            roles.append(v)
                        if "imdbID" in v or "Title" in v or "Poster" in v:
                            movies.append(v)

    # top-level keys fallback
    if not roles:
        if isinstance(crew_out.get("candidates"), list) or isinstance(crew_out.get("recommended_pool"), list):
            # wrap into a single default role
            roles = [{"role": crew_out.get("role") or "Lead", "candidates": crew_out.get("candidates") or crew_out.get("recommended_pool"), "movies": crew_out.get("movies") or []}]
        elif isinstance(crew_out.get("movies"), list):
            movies = crew_out.get("movies")

    # Extract from final output; also consider per-task outputs if provided under 'tasks_output'
    tasks = crew_out.get("tasks_output") or []
    if isinstance(tasks, list):
        for t in tasks:
            if not isinstance(t, dict):
                continue
            out = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
            if isinstance(out, str):
                parsed = try_extract_json_from_string(out)
                if parsed is not None:
                    out = parsed
            if isinstance(out, list):
                if out and isinstance(out[0], dict) and ("role" in out[0] or "candidates" in out[0]):
                    roles = out
                if out and isinstance(out[0], dict) and ("imdbID" in out[0] or "Title" in out[0] or "Poster" in out[0]):
                    movies = out
            if isinstance(out, dict):
                if "candidates" in out and isinstance(out["candidates"], list):
                    roles.append({"role": out.get("role") or out.get("name_hint") or "Lead", "candidates": out["candidates"], "movies": out.get("movies") or []})
                if "movies" in out and isinstance(out["movies"], list):
                    movies = out["movies"]

    return roles, movies

def parse_fee_to_int(value: Any) -> Optional[int]:
    """
    Accept numeric, string with digits, or range like '25,00,000-35,00,000' and return integer INR.
    If multiple numbers present, take midpoint. Returns None if not parseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value)
    numbers = re.findall(r"\d+", s)
    if not numbers:
        return None
    nums = [int(n) for n in numbers]
    if len(nums) == 1:
        return nums[0]
    # if many numbers, try to find a reasonably sized pair at the end (handles long urls with digits)
    # prefer first two or last two
    if len(nums) >= 2:
        first, last = nums[0], nums[-1]
        return int((first + last) / 2)
    return None

def compute_candidate_stats(candidate: Dict[str, Any], movies_for_role: List[Dict[str, Any]], usd_to_inr: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["name"] = pick_first(candidate, ["name", "actor", "actor_name", "title"], "Unknown")
    # candidate-level imdb
    c_imdb = pick_first(candidate, ["imdb_rating", "average_imdb_rating", "avg_imdb", "imdb"], None)
    out["avg_imdb"] = float(c_imdb) if (c_imdb is not None and str(c_imdb).replace('.', '').isdigit()) else None
    # match movies by provided titles or ids
    matched = []
    titles = candidate.get("relevant_movie_titles") or candidate.get("relevant_movies") or []
    ids = candidate.get("relevant_movie_ids") or candidate.get("movie_ids") or candidate.get("movies_ids") or []
    lc_titles = {t.lower(): t for t in (titles or []) if isinstance(t, str)}
    lc_ids = {str(i): i for i in (ids or []) if isinstance(i, (str, int))}
    for m in movies_for_role or []:
        if not isinstance(m, dict):
            continue
        mt = (m.get("Title") or m.get("title") or "").strip()
        mid = m.get("imdbID") or m.get("imdb_id") or m.get("movie_id") or m.get("id")
        matched_by_title = mt and (mt.lower() in lc_titles)
        matched_by_id = mid and (str(mid) in lc_ids)
        if not (matched_by_title or matched_by_id) and mt and lc_titles:
            for t_low in lc_titles.keys():
                if t_low in mt.lower() or mt.lower() in t_low:
                    matched_by_title = True
                    break
        if matched_by_title or matched_by_id:
            matched.append(m)
    # fallback: match by actor name presence in movie Actors field
    if not matched:
        nm = out["name"]
        if nm and movies_for_role:
            for m in movies_for_role:
                actors_str = (m.get("Actors") or m.get("actors") or "")
                if isinstance(actors_str, str) and nm.lower() in actors_str.lower():
                    matched.append(m)
    # compute avg imdb and avg box
    imdb_vals = []
    box_vals = []
    for mm in matched:
        try:
            mr = mm.get("imdbRating") or mm.get("imdb_rating") or mm.get("imdb")
            if mr is not None:
                imdb_vals.append(float(mr))
        except Exception:
            pass
        bi = mm.get("BoxOfficeINR") or mm.get("box_office_inr") or mm.get("box_office") or mm.get("boxOffice")
        if bi is None:
            # parse from strings if available
            bi = pick_first(mm.get("_raw") or mm, ["BoxOffice", "box_office"])
            if bi:
                parsed = parse_money_str_to_int(bi, usd_to_inr)
                if parsed:
                    box_vals.append(parsed)
        else:
            try:
                if isinstance(bi, (int, float)):
                    box_vals.append(int(bi))
                else:
                    digits = re.sub(r"[^\d]", "", str(bi))
                    if digits:
                        box_vals.append(int(digits))
            except Exception:
                pass
    if out["avg_imdb"] is None and imdb_vals:
        out["avg_imdb"] = sum(imdb_vals) / len(imdb_vals)
    out["avg_box_inr"] = int(sum(box_vals) / len(box_vals)) if box_vals else None
    out["matched_movies"] = matched
    # fee
    fee_raw = pick_first(candidate, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None)
    out["fee_inr"] = parse_fee_to_int(fee_raw)
    return out

# ---- Streamlit UI ----
st.set_page_config(page_title="Right Casting Choice AI", layout="wide")
st.title("🎬 Right Casting Choice AI")

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_similar = st.number_input("Number of similar movies", min_value=1, max_value=10, value=3)
    user_budget_ui = st.number_input("User budget (UI units - Cr for Bollywood, M for Hollywood)", min_value=0.0, value=100.0, step=0.5)
    usd_to_inr = float(st.number_input("USD→INR rate", min_value=10.0, max_value=200.0, value=83.0))
    industry = st.selectbox("Industry", options=["hollywood", "bollywood"], index=1)

if industry == "bollywood":
    multiplier = 10_000_000
else:
    multiplier = 1_000_000

plot = st.text_area("Movie plot", value="A righteous and fearless police officer takes on corruption and crime to restore justice in his city.", height=140)
run_btn = st.button("Run Crew Flow")

if not run_btn:
    st.info("Fill inputs and click Run Crew Flow to fetch candidates and similar movies.")
    st.stop()

user_budget_raw = int(user_budget_ui * multiplier)
inputs = {
    "plot": plot,
    "n_similar": int(n_similar),
    "usd_to_inr": float(usd_to_inr),
    "user_budget_inr": int(user_budget_raw) if industry == "bollywood" else None,
    "user_budget_usd": int(user_budget_raw) if industry == "hollywood" else None,
    "industry": industry,
}

# spinner while crew runs (no extra message after it completes)
with st.spinner("Running crew pipeline..."):
    crew_output = safe_crew_kickoff(inputs)

# Stop early with a friendly banner if rate-limited
if isinstance(crew_output, dict) and crew_output.get("error"):
    st.error("Gemini rate limit/quota hit. Please wait and retry.")
    st.caption(str(crew_output.get("error")))
    st.stop()

# parse roles & movies
roles_list, movies_list = extract_roles_and_movies_from_crew(crew_output)

# detect initial character list from task outputs if present (tasks_output)
def detect_characters_from_tasks(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    chars: List[Dict[str, Any]] = []
    tasks = out.get("tasks_output") or []
    if not isinstance(tasks, list):
        return chars
    best_len = 0
    best_list: List[Dict[str, Any]] = []
    for t in tasks:
        try:
            if not isinstance(t, dict):
                continue
            text = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
            parsed = try_extract_json_from_string(text) if isinstance(text, str) else text
            if isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], dict) and ("role" in parsed[0] or "name_hint" in parsed[0] or "gender" in parsed[0]):
                    curr = [c for c in parsed if isinstance(c, dict)]
                    if len(curr) > best_len:
                        best_len = len(curr)
                        best_list = curr
        except Exception:
            continue
    return best_list or chars

initial_chars = detect_characters_from_tasks(crew_output)
initial_detected_count = len(initial_chars)

def _get_char_name(c: Dict[str, Any]) -> Optional[str]:
    return pick_first(c, ["name_hint", "role", "name"], None)

expected_names: List[str] = []
initial_by_name: Dict[str, Dict[str, Any]] = {}
for c in initial_chars:
    nm = _get_char_name(c)
    if nm:
        if nm not in expected_names:
            expected_names.append(nm)
        initial_by_name[nm] = c

# Use final crew output + optional tasks_output hints

def _get_role_name(rb: Dict[str, Any]) -> Optional[str]:
    return pick_first(rb, ["name_hint", "nameHint", "role"], None)

def _names_missing(roles: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not expected_names:
        return []
    final_names = []
    if isinstance(roles, list):
        for rb in roles:
            if isinstance(rb, dict):
                rn = _get_role_name(rb)
                if rn:
                    final_names.append(rn)
    missing = [n for n in expected_names if n not in set(final_names)]
    return missing

returned_less_than_detected = False
max_retries = 1
retry_count = 0
missing_names = _names_missing(roles_list)
while initial_detected_count and retry_count < max_retries and (
    (isinstance(roles_list, list) and len(roles_list) < initial_detected_count) or (missing_names)
):
    retry_count += 1
    augmented_inputs = dict(inputs)
    with st.spinner(f"Retrying to include all {initial_detected_count} characters (attempt {retry_count}/{max_retries})..."):
        crew_output = safe_crew_kickoff(augmented_inputs)
        roles_list, movies_list = extract_roles_and_movies_from_crew(crew_output)
        missing_names = _names_missing(roles_list)

if initial_detected_count and (
    (isinstance(roles_list, list) and len(roles_list) < initial_detected_count) or (missing_names)
):
    returned_less_than_detected = True

# if roles empty, attempt recursive heuristic find
if not roles_list:
    def find_roles_recursive(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                s = obj[0]
                if ("name" in s and ("score" in s or "movies" in s)) or ("role" in s and "candidates" in s):
                    return obj
            for it in obj:
                r = find_roles_recursive(it)
                if r:
                    return r
        elif isinstance(obj, dict):
            for v in obj.values():
                r = find_roles_recursive(v)
                if r:
                    return r
        return None
    maybe = find_roles_recursive(crew_output)
    if maybe:
        roles_list = [{"role": "Lead", "name_hint": "Lead", "candidates": maybe, "movies": movies_list}]

# gather movies if empty
if not movies_list and roles_list:
    collected = []
    for r in roles_list:
        for m in (r.get("movies") or []):
            if isinstance(m, dict):
                collected.append(m)
    movies_list = collected

# dedupe movies by imdbID
seen = set()
dedup = []
for m in (movies_list or []):
    mid = m.get("imdbID") or m.get("imdb_id") or m.get("movie_id") or m.get("id")
    if mid:
        if mid in seen:
            continue
        seen.add(mid)
    dedup.append(m)
movies_list = dedup

# Show detected characters summary (count + traits)
if initial_detected_count:
    st.subheader("Detected Characters")
    det_rows = []
    for c in initial_chars:
        nm = _get_char_name(c) or "Unknown"
        gen = pick_first(c, ["gender", "sex"], "")
        age = pick_first(c, ["age_range", "estimated_age_at_peak_performance_for_role"], "")
        traits = c.get("traits") or c.get("notes") or []
        if isinstance(traits, list):
            traits_str = ", ".join(map(str, traits))
        else:
            traits_str = str(traits)
        det_rows.append({
            "Character": nm,
            "Gender": gen,
            "Age Range": age,
            "Traits": traits_str,
        })
    try:
        st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
    except Exception:
        st.write(det_rows)
    st.caption(f"Detected {initial_detected_count} character(s) from the extractor.")

# Tabs
tab_candidates, tab_movies = st.tabs(["📋 Candidate Pool", "🎞 Similar Movies & Posters"])

# Candidate Pool
with tab_candidates:
    st.header(f"Audition List ({industry.capitalize()})")
    if returned_less_than_detected:
        if initial_detected_count and isinstance(roles_list, list):
            final_names = [_get_role_name(rb) for rb in roles_list if isinstance(rb, dict)]
            final_names = [n for n in final_names if n]
            still_missing = [n for n in expected_names if n not in set(final_names)]
            if still_missing:
                st.warning(
                    f"Returned {len(roles_list)} of detected {initial_detected_count} characters; missing: {', '.join(still_missing)}. Showing best effort."
                )
            else:
                st.warning(
                    f"Returned {len(roles_list)} of detected {initial_detected_count} characters. Showing best effort."
                )
        else:
            st.warning("Final candidate list does not include all initially detected characters. Showing best effort.")
    # schema normalization helpers
    def normalize_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
        nc: Dict[str, Any] = {}
        nc["name"] = pick_first(c, ["name", "actor", "actor_name", "title"], "Unknown")
        # imdb rating
        ir = pick_first(c, ["imdb_rating", "average_imdb_rating", "avg_imdb", "imdb"], None)
        try:
            nc["imdb_rating"] = float(ir) if ir is not None else None
        except Exception:
            nc["imdb_rating"] = None
        # box office
        abo = pick_first(c, ["average_box_office_inr", "avg_box_office_inr", "avg_box_inr", "box_office_inr"], None)
        try:
            nc["average_box_office_inr"] = int(abo) if abo is not None else None
        except Exception:
            nc["average_box_office_inr"] = None
        # fee
        fee_raw = pick_first(c, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None)
        nc["implied_actor_fee_estimate"] = parse_fee_to_int(fee_raw)
        # source
        nc["fee_source"] = pick_first(c, ["fee_source", "notes", "note", "explanation"], "")
        # score
        sc = pick_first(c, ["score", "combined_score", "raw_score"], None)
        try:
            nc["score"] = float(sc) if sc is not None else None
        except Exception:
            nc["score"] = None
        # passthrough hints
        nc["gender"] = pick_first(c, ["gender", "sex"], None)
        nc["age_range"] = pick_first(c, ["estimated_age_at_peak_performance_for_role", "age_range"], None)
        return nc

    def normalize_role_block(rb: Dict[str, Any]) -> Dict[str, Any]:
        nr: Dict[str, Any] = {}
        nr["role"] = rb.get("role") or "Role"
        nr["name_hint"] = pick_first(rb, ["name_hint", "nameHint"], "")
        nr["age_range"] = pick_first(rb, ["age_range", "estimated_age_at_peak_performance_for_role"], "")
        nr["gender"] = rb.get("gender") or ""
        nr["traits"] = rb.get("traits") or rb.get("notes") or []
        # Enrich missing traits/gender/age from initial detection by name match
        nm_for_lookup = nr["name_hint"] or nr["role"]
        if nm_for_lookup in initial_by_name:
            src = initial_by_name[nm_for_lookup]
            if not nr["gender"]:
                nr["gender"] = pick_first(src, ["gender", "sex"], "") or nr["gender"]
            if not nr["age_range"]:
                nr["age_range"] = pick_first(src, ["age_range", "estimated_age_at_peak_performance_for_role"], "") or nr["age_range"]
            if not nr["traits"]:
                nr["traits"] = src.get("traits") or src.get("notes") or nr["traits"]
        # movies list passthrough
        mlist = rb.get("movies") or movies_list or []
        nr["movies"] = mlist if isinstance(mlist, list) else []
        # candidates
        cand_list = rb.get("candidates") or rb.get("recommended_pool") or []
        if not isinstance(cand_list, list):
            cand_list = []
        nr["candidates"] = [normalize_candidate(c) for c in cand_list if isinstance(c, dict)]
        # recommended pool = top 6 by score
        sorted_pool = sorted(nr["candidates"], key=lambda x: (x.get("score") or 0.0, - (x.get("implied_actor_fee_estimate") or 0)), reverse=True)
        nr["recommended_pool"] = sorted_pool[:6]
        return nr
    if not roles_list:
        st.info("No role/candidate data found in crew output. See raw output below.")
    else:
        # normalize schema for UI
        roles_list = [normalize_role_block(r) for r in roles_list if isinstance(r, dict)]
        grand_total_fee_inr = 0  # sum of selected top candidate fees per role
        grand_roles_with_selected = 0
        selected_actor_names: set[str] = set()  # prevent same actor across multiple roles

        def ensure_candidates(role_block: Dict[str, Any], movies_for_role: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            cand_list = role_block.get("candidates") or role_block.get("recommended_pool") or []
            if cand_list and isinstance(cand_list, list):
                return cand_list
            # Derive simple actor list from movie metadata
            actor_names = []
            for mv in movies_for_role or []:
                actors_str = mv.get("Actors") or mv.get("actors") or ""
                if isinstance(actors_str, str) and actors_str:
                    for raw in actors_str.split(","):
                        nm = raw.strip()
                        if nm and nm.lower() not in {a.lower() for a in actor_names}:
                            actor_names.append(nm)
                if len(actor_names) >= 5:
                    break
            derived = [{"name": nm} for nm in actor_names[:5]]
            # If still empty, try Serper Dev Tool to propose candidates
            if not derived:
                try:
                    from crewai_tools import SerperDevTool
                    role = role_block.get("role") or role_block.get("name_hint") or "lead"
                    gender = (role_block.get("gender") or "").strip()
                    age = (role_block.get("age_range") or "").strip()
                    industry_hint = "Bollywood" if industry == "bollywood" else "Hollywood"
                    query = f"best {industry_hint} actors for {role} role {gender} {age}"
                    serper = SerperDevTool()
                    results = serper.run(query)
                    text_blob = results if isinstance(results, str) else json.dumps(results)
                    possible = re.findall(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text_blob)
                    unique_names = []
                    for nm in possible:
                        if nm.lower() not in {a.lower() for a in unique_names}:
                            unique_names.append(nm)
                        if len(unique_names) >= 5:
                            break
                    for nm in unique_names:
                        derived.append({"name": nm})
                except Exception:
                    pass
            if not derived:
                derived = [{"name": "Placeholder Actor"}]
            return derived

        for role_block in roles_list:
            role_name = role_block.get("role") or "Role"
            name_hint = role_block.get("name_hint") or role_block.get("nameHint") or ""
            age_hint = role_block.get("age_range") or role_block.get("estimated_age_at_peak_performance_for_role") or ""
            gender_hint = role_block.get("gender") or ""
            traits = role_block.get("traits") or role_block.get("notes") or []

            # Build header including name_hint (if present)
            header = f"Role: {role_name}"
            if name_hint:
                header += f" — name hint: {name_hint}"
            if age_hint:
                header += f" ({age_hint})"
            if gender_hint:
                header += f" — {gender_hint.capitalize()}"

            with st.expander(header, expanded=False):
                if traits:
                    if isinstance(traits, list):
                        st.write(f"**Traits:** {', '.join(map(str, traits))}")
                    else:
                        st.write(f"**Traits / Notes:** {str(traits)}")

                movies_for_role = role_block.get("movies") or movies_list or []
                candidates = role_block.get("candidates") or ensure_candidates(role_block, movies_for_role)

                rows = []
                selected_candidate_fee_inr: Optional[int] = None
                selected_candidate_name: Optional[str] = None
                # UI now selects the first candidate returned by crew for the role

                def _low_trust_fee_source(src: str) -> bool:
                    if not src:
                        return False
                    s = src.lower()
                    return ("networth" in s) or ("net worth" in s) or ("celebritynetworth" in s)

                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    stats = compute_candidate_stats(c, movies_for_role, usd_to_inr)
                    name = stats.get("name")
                    # Build table rows
                    # gender/age no longer displayed in the table
                    gender = pick_first(c, ["gender", "sex"], "")
                    age_range = pick_first(c, ["estimated_age_at_peak_performance_for_role", "age_range"], "")
                    avg_imdb = stats.get("avg_imdb")
                    avg_bo_inr = stats.get("avg_box_inr")
                    score = pick_first(c, ["score", "combined_score", "raw_score"], None)
                    fee_inr = stats.get("fee_inr") if "fee_inr" in stats else None
                    if fee_inr is None:
                        # also attempt to parse directly from candidate top-level keys
                        fee_inr = parse_fee_to_int(pick_first(c, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None))
                    src_txt = pick_first(c, ["fee_source", "notes", "note", "explanation"], "") or ""
                    low_trust = _low_trust_fee_source(src_txt)
                    # fee for display only; selection uses first candidate logic
                    fee_display = ""
                    if isinstance(fee_inr, int):
                        if industry == "hollywood":
                            fee_usd = fee_inr / usd_to_inr
                            fee_display = f"${(fee_usd / 1_000_000):,.2f} M"
                        else:
                            fee_display = f"₹{(fee_inr / 10_000_000):,.2f} Cr"
                        # selection logic
                        # Selection: pick the first candidate from crew output for this role (skip duplicates across roles)
                        if selected_candidate_name is None:
                            if isinstance(name, str) and name.strip() and name.strip().lower() in {n.lower() for n in selected_actor_names}:
                                pass  # already taken in previous role; keep showing in table but not select
                            else:
                                selected_candidate_name = name
                                selected_candidate_fee_inr = fee_inr

                    # display average box office in proper currency
                    if isinstance(avg_bo_inr, (int, float)):
                        if industry == "hollywood":
                            avg_bo_display = f"${((avg_bo_inr / usd_to_inr) / 1_000_000):,.2f} M"
                        else:
                            avg_bo_display = f"₹{(avg_bo_inr / 10_000_000):,.2f} Cr"
                    else:
                        avg_bo_display = ""

                    note_src = src_txt
                    if low_trust:
                        note_src = f"{src_txt} [flagged: net-worth source ignored for fee]"
                    rows.append({
                        "Actor Name": name,
                        "Avg. BO": avg_bo_display,
                        "Avg. IMDB": (f"{avg_imdb:.1f}" if avg_imdb is not None else ""),
                        "Score": (f"{float(score):.4f}" if score is not None else ""),
                        "Implied Fee": fee_display,
                        "Notes / Source": note_src
                    })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.write("No presentable candidate data for this role.")

                if selected_candidate_fee_inr is not None:
                    if industry == "hollywood":
                        fee_usd = selected_candidate_fee_inr / usd_to_inr
                        st.markdown(f"**Selected Top Candidate:** {selected_candidate_name} — Fee: ${fee_usd/1_000_000:,.2f} M")
                        # track grand total in USD for Hollywood
                        grand_total_fee_inr += int(selected_candidate_fee_inr)  # keep raw INR for compatibility
                    else:
                        st.markdown(f"**Selected Top Candidate:** {selected_candidate_name} — Fee: ₹{selected_candidate_fee_inr/10_000_000:,.2f} Cr")
                        grand_total_fee_inr += selected_candidate_fee_inr
                    if isinstance(selected_candidate_name, str) and selected_candidate_name.strip():
                        selected_actor_names.add(selected_candidate_name.strip())
                else:
                    st.markdown("**Selected Top Candidate:** No fee data available; cost omitted.")
                grand_roles_with_selected += 1

        # grand summary
        st.markdown("---")
        if grand_roles_with_selected:
            if industry == "hollywood":
                total_usd = (grand_total_fee_inr / usd_to_inr)
                st.markdown(f"### Total cost (top candidate per role): **${total_usd/1_000_000:,.2f} M**")
                st.markdown(f"### Your budget: **${user_budget_raw/1_000_000:,.2f} M**")
                remaining_usd = user_budget_raw - total_usd
                rem_display = f"${remaining_usd/1_000_000:,.2f} M"
            else:
                st.markdown(f"### Total cost (top candidate per role): **₹{grand_total_fee_inr/10_000_000:,.2f} Cr**")
                st.markdown(f"### Your budget: **₹{user_budget_raw/10_000_000:,.2f} Cr**")
                remaining_usd = None
                remaining = user_budget_raw - grand_total_fee_inr
                rem_display = f"₹{remaining/10_000_000:,.2f} Cr"
            if industry == "hollywood":
                if remaining_usd < 0:
                    st.error(f"Budget exceeded. Remaining: {rem_display}")
                else:
                    st.success(f"Remaining budget after counted fees: {rem_display}")
            else:
                if remaining < 0:
                    st.error(f"Budget exceeded. Remaining: {rem_display}")
                else:
                    st.success(f"Remaining budget after counted fees: {rem_display}")
            st.caption("Note: UI selects the first candidate returned by the crew for each role. Duplicate actors across roles are prevented, and totals reflect the sum of those first-choice fees.")
        else:
            st.info("No actor fee fields were parsed; total cost unavailable. Ensure your budget_ranker returns numeric implied_actor_fee_estimate in INR.")

# Similar Movies tab
with tab_movies:
    st.header("Similar Movies & Posters")
    if not movies_list:
        st.info("No similar movies were returned by the crew.")
    else:
        posters: List[Dict[str, str]] = []
        rows = []
        seen_posters = set()
        for m in movies_list:
            if not isinstance(m, dict):
                continue
            poster = pick_first(m, ["Poster", "poster", "poster_url", "PosterUrl"]) or pick_first(m.get("_raw") or {}, ["Poster", "poster"])
            title = pick_first(m, ["Title", "title", "name"]) or ""
            year = pick_first(m, ["Year", "year"]) or ""
            imdb = pick_first(m, ["imdbRating", "imdb_rating", "imdb"]) or ""
            box_inr, _ = normalize_movie_boxoffice(m, usd_to_inr)
            # add poster dedup
            if poster and poster not in seen_posters:
                posters.append({"poster": poster, "title": title})
                seen_posters.add(poster)
            rows.append({
                "Title": title,
                "Year": year,
                "IMDB": imdb,
                "BoxOffice (INR)" if industry == "bollywood" else "BoxOffice (USD)": (box_inr if industry == "bollywood" else (int(box_inr / usd_to_inr) if box_inr else None))
            })
        # posters row
        if posters:
            cols = st.columns(min(len(posters), n_similar))
            for i, p in enumerate(posters[:n_similar]):
                with cols[i]:
                    st.image(p["poster"], use_column_width=True)
                    st.caption(p["title"])
        # movie table
        try:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception:
            st.write(rows)


# debugging raw output
st.subheader("Raw Crew Output (parsed)")
st.json(crew_output)
if isinstance(crew_output, dict):
    st.caption(f"Top-level keys: {list(crew_output.keys())}")

# Print each task result if tasks_output provided
tasks_out = crew_output.get("tasks_output") or []
if isinstance(tasks_out, list) and tasks_out:
    st.subheader("Task Outputs")
    for i, t in enumerate(tasks_out, start=1):
        st.markdown(f"**Task {i}:** {t.get('name') or t.get('id') or ''}")
        raw_out = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
        if isinstance(raw_out, str):
            parsed = try_extract_json_from_string(raw_out)
            st.json(parsed if parsed is not None else raw_out)
        else:
            st.json(raw_out)

st.success("Done — candidate pool and similar movies displayed.")
