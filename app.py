# app.py (updated: spinner while crew runs + Candidate Pool + Similar Movies tab)
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load env & make src importable
load_dotenv()
sys.path.append(str((Path(__file__).parent / "src").resolve()))

# Crew imports (keep as in your project)
from right_casting_choice_ai.crew import RightCastingChoiceAi
from right_casting_choice_ai.tools.omdb import OmdbTool

# Minimal guard for Gemini import (unused here)
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    genai = None
    _HAS_GEMINI = False


# -------------------- Utilities --------------------
def try_extract_json_from_string(s: Optional[str]) -> Optional[Any]:
    """Attempt to extract JSON object/list from a string (handles ```json fences)."""
    if s is None:
        return None
    txt = s.strip()
    # remove fenced blocks
    txt = re.sub(r"^```json\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"^```\s*", "", txt)
    txt = re.sub(r"```\s*$", "", txt)
    # try direct
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try find first JSON-like substring
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', txt)
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


def safe_crew_kickoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run crew.kickoff once and attempt to return a dict. Never raise."""
    try:
        crew = RightCastingChoiceAi().crew()
        result = crew.kickoff(inputs=inputs)
        # Prefer to_dict
        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass
        # raw attribute
        if hasattr(result, "raw"):
            raw = result.raw
            # if dict/list return as-is
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, list):
                return {"raw": raw}
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                if parsed is not None:
                    # if parsed is list/dict return inside {"raw": parsed} so UI knows it's raw-block
                    return {"raw": parsed, "verbose_raw": raw}
                return {"raw": raw}
        # try dict(result)
        try:
            d = dict(result)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        # fallback: string repr
        s = str(result)
        parsed = try_extract_json_from_string(s)
        if parsed is not None:
            return {"raw": parsed, "verbose_raw": s}
        return {"raw": s}
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}


def is_indian_movie(m: Dict[str, Any]) -> bool:
    """Simple heuristic - used earlier in your app. Keep it minimal here."""
    if not isinstance(m, dict):
        return False
    raw = m.get("_raw") or m.get("raw") or {}
    if isinstance(raw, dict):
        lang = (raw.get("Language") or raw.get("language") or "").lower()
        country = (raw.get("Country") or raw.get("country") or "").lower()
        if "india" in country or "hindi" in lang:
            return True
    title = (m.get("Title") or m.get("title") or "").lower()
    return any(tok in title for tok in ["raj", "bahubali", "gully", "raazi", "padma", "tanhaji", "drishyam"])


def normalize_movie_boxoffice(m: Dict[str, Any], usd_to_inr: float) -> Tuple[Optional[int], Optional[int]]:
    """Return (box_inr, budget_inr) attempting to read common fields."""
    def parse_money_str(s):
        if s is None:
            return None
        s = str(s)
        s = s.strip()
        if not s or s.upper() == "N/A":
            return None
        digits = re.sub(r"[^0-9]", "", s)
        if not digits:
            return None
        try:
            return int(digits)
        except Exception:
            return None

    box_inr = None
    if isinstance(m.get("BoxOfficeINR"), (int, float)):
        box_inr = int(m.get("BoxOfficeINR"))
    elif isinstance(m.get("box_office_inr"), (int, float)):
        box_inr = int(m.get("box_office_inr"))
    else:
        # try BoxOffice or box_office string
        raw = m.get("BoxOffice") or m.get("box_office")
        parsed = parse_money_str(raw)
        if parsed:
            # assume USD unless Indian movie
            if is_indian_movie(m):
                box_inr = parsed
            else:
                box_inr = int(parsed * usd_to_inr)

    budget_inr = None
    if isinstance(m.get("BudgetINR"), (int, float)):
        budget_inr = int(m.get("BudgetINR"))
    elif isinstance(m.get("budget_inr"), (int, float)):
        budget_inr = int(m.get("budget_inr"))
    else:
        raw = m.get("Budget") or m.get("budget")
        parsed = parse_money_str(raw)
        if parsed:
            if is_indian_movie(m):
                budget_inr = parsed
            else:
                budget_inr = int(parsed * usd_to_inr)
    return box_inr, budget_inr


# new helpers to robustly read candidate fields
def get_first(d: Dict[str, Any], keys: List[str], default=None):
    """Return first non-None value from d for any of keys (supports nested 'raw')."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    # check 'raw' nested dict
    raw = d.get("raw")
    if isinstance(raw, dict):
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
    return default


def compute_avg_bo_from_movies(candidate: Dict[str, Any], role_movies: List[Dict[str, Any]], usd_to_inr: float) -> Optional[int]:
    """If candidate lacks avg box office, try to compute average from role_movies where actor appears."""
    if not isinstance(role_movies, list):
        return None
    total = 0
    count = 0
    name = get_first(candidate, ["name", "actor", "title"]) or ""
    for m in role_movies:
        if not isinstance(m, dict):
            continue
        actors_str = (m.get("Actors") or m.get("actors") or "")
        try:
            if name and isinstance(actors_str, str) and name.lower() in actors_str.lower():
                bo, _ = normalize_movie_boxoffice(m, usd_to_inr)
                if bo:
                    total += bo
                    count += 1
        except Exception:
            continue
    if count == 0:
        return None
    return int(total / count)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Right Casting Choice AI — Final", layout="wide")
st.title("🎬 Right Casting Choice AI — Final")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    n_similar = st.number_input("Number of similar movies", min_value=1, max_value=10, value=3)
    user_budget_ui = st.number_input(
        "User budget (UI units - Cr for Bollywood, M for Hollywood)", min_value=0.0, value=100.0, step=0.5
    )
    usd_to_inr = float(st.number_input("USD→INR rate", min_value=10.0, max_value=200.0, value=83.0))
    industry = st.selectbox("Industry", options=["hollywood", "bollywood"], index=1)

# display units
if industry == "bollywood":
    currency_symbol = "₹"
    unit_label = "Cr"
    multiplier = 10_000_000
else:
    currency_symbol = "$"
    unit_label = "M"
    multiplier = 1_000_000

plot = st.text_area("Movie plot", value="A righteous and fearless police officer takes on corruption and crime to restore justice in his city.", height=150)
run_btn = st.button("Run Crew Flow")

if not run_btn:
    st.info("Fill inputs and click Run Crew Flow to fetch candidates and similar movies.")
    st.stop()

# Build inputs & call crew once
user_budget_raw = int(user_budget_ui * multiplier)
inputs = {
    "plot": plot,
    "n_similar": int(n_similar),
    "usd_to_inr": float(usd_to_inr),
    "user_budget_inr": int(user_budget_raw) if industry == "bollywood" else None,
    "user_budget_usd": int(user_budget_raw) if industry == "hollywood" else None,
    "industry": industry,
}

# ------------------ Simple spinner while crew runs ------------------
st.info("Running crew.kickoff() — one-shot run (may take a few seconds).")
with st.spinner("Running crew pipeline..."):
    crew_output = safe_crew_kickoff(inputs)

# ---------------- Try to find the normalized data the crew returned. ----------------
def extract_roles_and_movies_from_crew(crew_out: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (roles_list, movies_list)
    roles_list is a list of role dicts each containing 'role'/'name_hint'/'candidates'/'movies' etc.
    movies_list is list of movies metadata dicts (with Poster etc).
    """
    roles = []
    movies = []

    # direct top-level
    if isinstance(crew_out, dict):
        # Case 1: crew_out already has top-level structure
        if crew_out.get("raw") and isinstance(crew_out["raw"], list):
            candidate_block = crew_out["raw"]
            if candidate_block and isinstance(candidate_block[0], dict) and ("role" in candidate_block[0] or "candidates" in candidate_block[0]):
                roles = candidate_block
            else:
                # search for role-like dicts and movie-like dicts inside the list
                for v in candidate_block:
                    if isinstance(v, dict):
                        if "role" in v or "candidates" in v or "name_hint" in v:
                            roles.append(v)
                        if "imdbID" in v or "Title" in v or "title" in v or "Poster" in v:
                            movies.append(v)
        elif crew_out.get("raw") and isinstance(crew_out["raw"], str):
            parsed = try_extract_json_from_string(crew_out["raw"])
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict) and ("role" in parsed[0] or "candidates" in parsed[0]):
                    roles = parsed
                else:
                    for v in parsed:
                        if isinstance(v, dict):
                            if "imdbID" in v or "Title" in v or "title" in v:
                                movies.append(v)
                            if "role" in v or "candidates" in v:
                                roles.append(v)
        # direct top-level keys
        if not roles:
            if isinstance(crew_out.get("candidates"), list) and isinstance(crew_out.get("recommended_pool"), list):
                roles = [{"role": "Lead", "name_hint": "Lead", "candidates": crew_out.get("candidates"), "recommended_pool": crew_out.get("recommended_pool")}]
            elif isinstance(crew_out.get("recommended_pool"), list) and isinstance(crew_out.get("movies"), list):
                roles = [{"role": "Lead", "name_hint": "Lead", "candidates": crew_out.get("recommended_pool"), "movies": crew_out.get("movies")}]
            elif isinstance(crew_out.get("movies"), list):
                movies = crew_out.get("movies")
            # check tasks outputs for extract_characters_task or rank_candidates_task
            tasks = crew_out.get("tasks") or []
            if isinstance(tasks, list) and tasks:
                for t in tasks:
                    if not isinstance(t, dict):
                        continue
                    out = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
                    if isinstance(out, str):
                        parsed = try_extract_json_from_string(out)
                        if parsed is not None:
                            out = parsed
                    if isinstance(out, list) and out:
                        if isinstance(out[0], dict) and ("role" in out[0] or "candidates" in out[0] or "name_hint" in out[0]):
                            roles = out
                        if isinstance(out[0], dict) and ("imdbID" in out[0] or "Title" in out[0] or "poster" in out[0]):
                            movies = out
                    if isinstance(out, dict):
                        if "candidates" in out and isinstance(out["candidates"], list):
                            role_name = out.get("role") or out.get("name_hint") or "Lead"
                            roles.append({"role": role_name, "candidates": out.get("candidates"), "movies": out.get("movies") or []})
                        if "movies" in out and isinstance(out["movies"], list):
                            movies = out.get("movies")
    return roles, movies


roles_list, movies_list = extract_roles_and_movies_from_crew(crew_output)

# If roles_list is empty, do a last-ditch recursive search for anything that looks like candidate lists
if not roles_list:
    def find_roles_recursive(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                sample = obj[0]
                if ("name" in sample and ("score" in sample or "movies" in sample)) or ("role" in sample and "candidates" in sample):
                    return obj
            for item in obj:
                res = find_roles_recursive(item)
                if res:
                    return res
        elif isinstance(obj, dict):
            for v in obj.values():
                res = find_roles_recursive(v)
                if res:
                    return res
        return None
    maybe = find_roles_recursive(crew_output)
    if maybe:
        if isinstance(maybe, list) and maybe and isinstance(maybe[0], dict) and "name" in maybe[0]:
            roles_list = [{"role": "Lead", "name_hint": "Lead", "candidates": maybe, "movies": movies_list}]
        else:
            roles_list = []

# Normalize movies_list: if movies_list is empty but roles_list have 'movies' use them
if not movies_list and roles_list:
    collected = []
    for r in roles_list:
        for m in (r.get("movies") or []):
            if isinstance(m, dict):
                collected.append(m)
    movies_list = collected

# Create tabs: Candidate Pool and Similar Movies (as requested)
tab_candidates, tab_movies = st.tabs(["📋 Candidate Pool", "🎞 Similar Movies & Posters"])

# ----------------------- Candidate Pool Tab -----------------------
with tab_candidates:
    st.header(f"Audition List ({industry.capitalize()})")
    if not roles_list:
        st.info("No role/candidate data could be found in crew output. Check crew logs or task outputs.")
    else:
        # show all roles as expanders like your screenshot
        for role_block in roles_list:
            role_name = role_block.get("role") or role_block.get("name_hint") or "Role"
            age_hint = role_block.get("age_range") or role_block.get("estimated_age_at_peak_performance_for_role") or ""
            gender = role_block.get("gender") or ""
            traits = role_block.get("traits") or role_block.get("notes") or []
            # build descriptive header
            header = f"Role: {role_name}"
            if age_hint:
                header += f" ({age_hint})"
            if gender:
                header += f" — {gender.capitalize()}"
            with st.expander(header, expanded=False):
                if traits:
                    # traits might be list or string
                    if isinstance(traits, list):
                        st.write(f"**Traits:** {', '.join(map(str, traits))}")
                    else:
                        st.write(f"**Traits / Notes:** {str(traits)}")
                candidates = role_block.get("candidates") or role_block.get("recommended_pool") or []
                role_movies = role_block.get("movies") or movies_list or []

                if not candidates:
                    st.write("No actors found for this role.")
                    continue

                # Build display table
                rows = []
                for c in candidates:
                    # candidate object formats vary; try to extract sensible fields
                    name = get_first(c, ["name", "actor", "title"]) or "Unknown"
                    gender_val = get_first(c, ["gender", "sex"]) or ""
                    age_range = get_first(c, ["estimated_age_at_peak_performance_for_role", "age_range"]) or ""
                    avg_imdb = get_first(c, ["average_imdb_rating", "average_imdb", "avg_imdb", "imdb_rating", "avg_imdb_rating", "score"])
                    # try to make float
                    avg_imdb_f = None
                    try:
                        if avg_imdb is not None:
                            avg_imdb_f = float(avg_imdb)
                            # some crews normalize to 0..1; if avg_imdb <=1 and >0 assume scaled and convert to 10-scale
                            if 0 < avg_imdb_f <= 1.0:
                                avg_imdb_f = avg_imdb_f * 10.0
                    except Exception:
                        avg_imdb_f = None

                    avg_bo = get_first(c, ["average_box_office_inr", "avg_box_office_inr", "average_box_office", "avg_box_office", "average_box_office_usd"])
                    # If avg_bo missing, try compute from role_movies
                    if avg_bo is None:
                        avg_bo = compute_avg_bo_from_movies(c, role_movies, usd_to_inr)

                    score = get_first(c, ["score", "combined_score", "raw_score"])
                    notes = get_first(c, ["notes", "note", "notes_summary"]) or ""

                    # Convert INR -> UI display units if possible
                    bo_display = ""
                    if isinstance(avg_bo, (int, float)):
                        bo_display = f"{currency_symbol}{(avg_bo / multiplier):,.2f} {unit_label}"

                    rows.append({
                        "Actor Name": name,
                        "Gender": gender_val,
                        "Est. Age": age_range,
                        f"Avg. BO ({unit_label})": bo_display,
                        "Avg. IMDB": (f"{avg_imdb_f:.1f}" if avg_imdb_f is not None else ""),
                        "Score": (f"{float(score):.4f}" if score is not None else ""),
                        "Notes": notes
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

# ----------------------- Similar Movies Tab -----------------------
with tab_movies:
    st.header("Similar Movies & Posters")
    if not movies_list:
        st.info("No similar movies were returned by the crew.")
    else:
        # Show posters in a row up to n_similar
        posters = []
        rows = []
        for m in movies_list:
            if not isinstance(m, dict):
                continue
            poster = m.get("Poster") or m.get("poster") or m.get("poster_url") or ""
            title = m.get("Title") or m.get("title") or m.get("name") or ""
            year = m.get("Year") or m.get("year") or ""
            imdb = m.get("imdbRating") or m.get("imdb_rating") or ""
            # inside Similar Movies tab
            box_inr, budget_inr = normalize_movie_boxoffice(m, usd_to_inr)
            box_usd = int(box_inr / usd_to_inr) if box_inr else None
            budget_usd = int(budget_inr / usd_to_inr) if budget_inr else None

            rows.append({
                "Title": title,
                "Year": year,
                "IMDB": imdb,
                "BoxOffice (INR)" if industry=="bollywood" else "BoxOffice (USD)":
                    box_inr if industry=="bollywood" else box_usd
            })

            if poster:
                posters.append({"poster": poster, "title": title})
            if len(posters) >= n_similar:
                # keep only n_similar posters
                posters = posters[:n_similar]
                break

        # posters row
        if posters:
            cols = st.columns(min(len(posters), n_similar))
            for i, p in enumerate(posters[:n_similar]):
                with cols[i]:
                    st.image(p["poster"], use_column_width=True)
                    st.caption(p["title"])

        # movie table
        try:
            df_movies = pd.DataFrame(rows)
            st.dataframe(df_movies, use_container_width=True)
        except Exception:
            st.write(rows)

# Present raw JSON and top-level keys for debugging
st.subheader("Raw Crew Output (parsed)")
st.json(crew_output)
if isinstance(crew_output, dict):
    st.caption(f"Top-level keys: {list(crew_output.keys())}")
st.success("Done — candidate pool and similar movies displayed.")
