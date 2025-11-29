# app.py (robust parsing + candidate stats from movies)
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

# Project imports
from right_casting_choice_ai.crew import RightCastingChoiceAi
from right_casting_choice_ai.tools.omdb import OmdbTool

# Minimal Gemini guard (unused)
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    genai = None
    _HAS_GEMINI = False


# -------------------- Utilities --------------------
def try_extract_json_from_string(s: Optional[str]) -> Optional[Any]:
    if s is None:
        return None
    txt = str(s).strip()
    txt = re.sub(r"^```json\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"^```\s*", "", txt)
    txt = re.sub(r"```\s*$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        pass
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
    try:
        crew = RightCastingChoiceAi().crew()
        result = crew.kickoff(inputs=inputs)
        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass
        if hasattr(result, "raw"):
            raw = result.raw
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, list):
                return {"raw": raw}
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                return {"raw": parsed} if parsed is not None else {"raw": raw}
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
        return {"raw": s}
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}


def is_indian_movie(m: Dict[str, Any]) -> bool:
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
    def parse_money_str(s):
        if s is None:
            return None
        s = str(s).strip()
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
        raw = m.get("BoxOffice") or m.get("box_office") or m.get("BoxOfficeUSD") or m.get("boxOffice")
        parsed = parse_money_str(raw)
        if parsed:
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


def pick_first(obj: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return v
    return default


def compute_candidate_stats(candidate: Dict[str, Any], movies_for_role: List[Dict[str, Any]], usd_to_inr: float) -> Dict[str, Any]:
    """
    Build helpful stats for the candidate:
    - avg_imdb (prefer candidate.imdb_rating then movie imdbRatings)
    - avg_box_office_inr (average over matched movies with BoxOfficeINR)
    - matched_movies (list of matched movie dicts)
    """
    out: Dict[str, Any] = {}
    # candidate-level fields
    out["name"] = pick_first(candidate, ["name", "actor", "actor_name", "title"], "Unknown")
    out["score"] = pick_first(candidate, ["score", "combined_score", "raw_score", "popularity_score"], None)
    # Candidate may include imdb_rating (your crew sometimes provides this)
    c_imdb = pick_first(candidate, ["imdb_rating", "average_imdb_rating", "avg_imdb", "imdb"], None)
    if c_imdb is not None:
        try:
            out["avg_imdb"] = float(c_imdb)
        except Exception:
            out["avg_imdb"] = None
    else:
        out["avg_imdb"] = None

    # Try to match candidate -> movies
    matched = []
    # candidate might list relevant_movie_titles or movie ids
    titles = candidate.get("relevant_movie_titles") or candidate.get("movies") or candidate.get("relevant_movies") or []
    ids = candidate.get("relevant_movie_ids") or candidate.get("movie_ids") or candidate.get("movies_ids") or []
    # Normalize titles to lowercase for matching
    lc_titles = {t.lower(): t for t in (titles or []) if isinstance(t, str)}
    lc_ids = {i: i for i in (ids or []) if isinstance(i, str) or isinstance(i, int)}

    for m in movies_for_role or []:
        if not isinstance(m, dict):
            continue
        m_title = (m.get("Title") or m.get("title") or "").strip()
        m_id = m.get("imdbID") or m.get("imdb_id") or m.get("movie_id") or m.get("id")
        matched_by_title = m_title and (m_title.lower() in lc_titles)
        matched_by_id = m_id and (str(m_id) in lc_ids)
        # also match by substring if no direct exact match (helps with small title variants)
        if not (matched_by_title or matched_by_id) and m_title and lc_titles:
            for t_low in lc_titles.keys():
                if t_low in m_title.lower() or m_title.lower() in t_low:
                    matched_by_title = True
                    break
        if matched_by_title or matched_by_id:
            matched.append(m)

    # If no explicit candidate->movie map provided, attempt to assign movies by actor appearing in movie Actors field
    if not matched:
        cand_name = out["name"]
        if cand_name and movies_for_role:
            for m in movies_for_role:
                actors_str = (m.get("Actors") or m.get("actors") or "")
                if isinstance(actors_str, str) and cand_name.lower() in actors_str.lower():
                    matched.append(m)

    # Compute average imdb & average box office INR from matched movies if candidate-level avg_imdb missing
    imdb_vals = []
    box_vals = []
    for mm in matched:
        try:
            mr = mm.get("imdbRating") or mm.get("imdb_rating") or mm.get("imdb")
            if mr is not None:
                imdb_vals.append(float(mr))
        except Exception:
            pass
        # BoxOffice numeric fields already in INR in your crew output; normalize
        bi = mm.get("BoxOfficeINR") or mm.get("box_office_inr") or mm.get("box_office") or mm.get("boxOffice")
        # if BoxOfficeINR is None, try parsing BoxOffice string
        if bi is None:
            # leave None
            pass
        else:
            try:
                if isinstance(bi, (int, float)):
                    box_vals.append(int(bi))
                else:
                    # try to parse numeric string
                    digits = re.sub(r"[^0-9]", "", str(bi))
                    if digits:
                        box_vals.append(int(digits))
            except Exception:
                pass

    # average imdb fallback
    if out["avg_imdb"] is None and imdb_vals:
        try:
            out["avg_imdb"] = sum(imdb_vals) / len(imdb_vals)
        except Exception:
            out["avg_imdb"] = None

    # avg box office
    if box_vals:
        try:
            out["avg_box_inr"] = int(sum(box_vals) / len(box_vals))
        except Exception:
            out["avg_box_inr"] = None
    else:
        out["avg_box_inr"] = None

    out["matched_movies"] = matched
    return out


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Right Casting Choice AI — Final", layout="wide")
st.title("🎬 Right Casting Choice AI — Final")

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_similar = st.number_input("Number of similar movies", min_value=1, max_value=10, value=3)
    user_budget_ui = st.number_input("User budget (UI units - Cr for Bollywood, M for Hollywood)", min_value=0.0, value=100.0, step=0.5)
    usd_to_inr = float(st.number_input("USD→INR rate", min_value=10.0, max_value=200.0, value=83.0))
    industry = st.selectbox("Industry", options=["hollywood", "bollywood"], index=1)

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

user_budget_raw = int(user_budget_ui * multiplier)
inputs = {
    "plot": plot,
    "n_similar": int(n_similar),
    "usd_to_inr": float(usd_to_inr),
    "user_budget_inr": int(user_budget_raw) if industry == "bollywood" else None,
    "user_budget_usd": int(user_budget_raw) if industry == "hollywood" else None,
    "industry": industry,
}

# spinner while crew runs
with st.spinner("Running crew pipeline..."):
    crew_output = safe_crew_kickoff(inputs)

# ---------- parse roles and movies ----------
def extract_roles_and_movies_from_crew(crew_out: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    roles = []
    movies = []
    if isinstance(crew_out, dict):
        raw = crew_out.get("raw")
        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict) and ("role" in raw[0] or "candidates" in raw[0] or "name_hint" in raw[0]):
                roles = raw
            else:
                for v in raw:
                    if isinstance(v, dict):
                        if "role" in v or "candidates" in v or "name_hint" in v:
                            roles.append(v)
                        if "imdbID" in v or "Title" in v or "poster" in v:
                            movies.append(v)
        elif isinstance(raw, dict):
            if "movies" in raw and isinstance(raw["movies"], list):
                movies = raw["movies"]
            if "candidates" in raw and isinstance(raw["candidates"], list):
                roles = [{"role": raw.get("role") or raw.get("name_hint") or "Lead", "candidates": raw["candidates"], "movies": raw.get("movies", [])}]
        elif isinstance(raw, str):
            parsed = try_extract_json_from_string(raw)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict) and ("role" in parsed[0] or "candidates" in parsed[0] or "name_hint" in parsed[0]):
                    roles = parsed
                else:
                    for v in parsed:
                        if isinstance(v, dict):
                            if "role" in v or "candidates" in v or "name_hint" in v:
                                roles.append(v)
                            if "imdbID" in v or "Title" in v or "poster" in v:
                                movies.append(v)

        if not roles:
            if isinstance(crew_out.get("candidates"), list) and isinstance(crew_out.get("recommended_pool"), list):
                roles = [{"role": "Lead", "name_hint": "Lead", "candidates": crew_out.get("candidates"), "recommended_pool": crew_out.get("recommended_pool")}]
            elif isinstance(crew_out.get("recommended_pool"), list) and isinstance(crew_out.get("movies"), list):
                roles = [{"role": "Lead", "name_hint": "Lead", "candidates": crew_out.get("recommended_pool"), "movies": crew_out.get("movies")}]
            elif isinstance(crew_out.get("movies"), list) and not movies:
                movies = crew_out.get("movies")

        tasks = crew_out.get("tasks") or []
        if isinstance(tasks, list):
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
                        roles.append({"role": role_name, "candidates": out["candidates"], "movies": out.get("movies") or []})
                    if "movies" in out and isinstance(out["movies"], list):
                        movies = out["movies"]
    return roles, movies


roles_list, movies_list = extract_roles_and_movies_from_crew(crew_output)

# last-resort find roles
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

# If movies_list empty but roles contain movies, collect
if not movies_list and roles_list:
    collected = []
    for r in roles_list:
        for m in (r.get("movies") or []):
            if isinstance(m, dict):
                collected.append(m)
    movies_list = collected

# dedupe movies by imdbID
seen = set()
dedup_movies = []
for m in (movies_list or []):
    mid = m.get("imdbID") or m.get("imdb_id") or m.get("movie_id") or m.get("id")
    if mid:
        if mid in seen:
            continue
        seen.add(mid)
    dedup_movies.append(m)
movies_list = dedup_movies

# Tabs
tab_candidates, tab_movies = st.tabs(["📋 Candidate Pool", "🎞 Similar Movies & Posters"])

# Candidate Pool tab
with tab_candidates:
    st.header(f"Audition List ({industry.capitalize()})")
    if not roles_list:
        st.info("No role/candidate data could be found in crew output. Check crew logs or task outputs.")
    else:
        for role_block in roles_list:
            role_name = role_block.get("role") or role_block.get("name_hint") or "Role"
            age_hint = role_block.get("age_range") or role_block.get("estimated_age_at_peak_performance_for_role") or ""
            gender_hint = role_block.get("gender") or ""
            traits = role_block.get("traits") or role_block.get("notes") or []
            header = f"Role: {role_name}"
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

                candidates = role_block.get("candidates") or role_block.get("recommended_pool") or []
                movies_for_role = role_block.get("movies") or movies_list or []

                if not candidates:
                    st.write("No actors found for this role.")
                    continue

                # Build rows
                rows = []
                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    stats = compute_candidate_stats(c, movies_for_role, usd_to_inr)
                    name = stats.get("name")
                    gender = pick_first(c, ["gender", "sex"], "")
                    age_range = pick_first(c, ["estimated_age_at_peak_performance_for_role", "age_range"], "")
                    avg_imdb = stats.get("avg_imdb")
                    avg_bo_inr = stats.get("avg_box_inr")
                    score = stats.get("score")
                    # format display
                    bo_display = ""
                    if isinstance(avg_bo_inr, (int, float)):
                        bo_display = f"{currency_symbol}{(avg_bo_inr / multiplier):,.2f} {unit_label}"
                    rows.append({
                        "Actor Name": name,
                        "Gender": gender,
                        "Est. Age": age_range,
                        f"Avg. BO ({unit_label})": bo_display,
                        "Avg. IMDB": (f"{avg_imdb:.1f}" if avg_imdb is not None else ""),
                        "Score": (f"{float(score):.4f}" if score is not None else ""),
                        "Notes": pick_first(c, ["notes", "note", "explanation"], ""),
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No presentable candidate data for this role.")

# Similar Movies tab
with tab_movies:
    st.header("Similar Movies & Posters")
    if not movies_list:
        st.info("No similar movies were returned by the crew.")
    else:
        posters = []
        rows = []
        seen_posters = set()
        for m in movies_list:
            if not isinstance(m, dict):
                continue
            poster = pick_first(m, ["Poster", "poster", "poster_url", "PosterUrl"], "")
            title = pick_first(m, ["Title", "title", "name"], "")
            year = pick_first(m, ["Year", "year"], "")
            imdb = pick_first(m, ["imdbRating", "imdb_rating", "imdb"], "")
            box_inr, _ = normalize_movie_boxoffice(m, usd_to_inr)
            if poster and poster not in seen_posters:
                posters.append({"poster": poster, "title": title})
                seen_posters.add(poster)
            rows.append({
                "Title": title,
                "Year": year,
                "IMDB": imdb,
                "BoxOffice (INR)" if industry == "bollywood" else "BoxOffice (USD)": (box_inr if industry == "bollywood" else (int(box_inr / usd_to_inr) if box_inr else None))
            })
        # show posters (top n_similar)
        if posters:
            cols = st.columns(min(len(posters), n_similar))
            for i, p in enumerate(posters[:n_similar]):
                with cols[i]:
                    st.image(p["poster"], use_column_width=True)
                    st.caption(p["title"])
        try:
            df_movies = pd.DataFrame(rows)
            st.dataframe(df_movies, use_container_width=True)
        except Exception:
            st.write(rows)

# Show raw parsed JSON for debugging
st.subheader("Raw Crew Output (parsed)")
st.json(crew_output)
if isinstance(crew_output, dict):
    st.caption(f"Top-level keys: {list(crew_output.keys())}")

st.success("Done — candidate pool and similar movies displayed.")
