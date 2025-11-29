# app.py (final - Candidate Pool + Similar Movies tab)
"""
Right Casting Choice AI - Streamlit App (final)

- Candidate Pool tab: actors per extracted character (tables + posters)
- Similar Movies tab: prioritized posters and movie table
- Robust parsing of crew.kickoff outputs (handles JSON-in-strings and many shapes)
- Industry-aware units (Cr for Bollywood, M for Hollywood)
"""
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

# Load .env and normalize keys
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

# Ensure src is importable (adjust if needed)
sys.path.append(str((Path(__file__).parent / "src").resolve()))

# Import crew class (your project)
from right_casting_choice_ai.crew import RightCastingChoiceAi

# ---------- Helpers ----------
def try_extract_json_from_string(s: Optional[str]) -> Optional[Any]:
    """Try to extract JSON from a string (handles fenced ```json blocks)."""
    if s is None:
        return None
    text = str(s).strip()
    # remove backticks / fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first {...} or [...]
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if m:
        cand = m.group(1)
        try:
            return json.loads(cand)
        except Exception:
            # last resort single->double quote replacement
            try:
                return json.loads(cand.replace("'", '"'))
            except Exception:
                return None
    return None


def safe_crew_kickoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one crew.kickoff and return a dict-like object.
    Tries many fallbacks to produce a usable dictionary.
    """
    try:
        crew = RightCastingChoiceAi().crew()
        result = crew.kickoff(inputs=inputs)
        # prefer to_dict
        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass
        # check raw attribute
        if hasattr(result, "raw"):
            raw = result.raw
            if isinstance(raw, dict) and raw:
                return raw
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                if parsed:
                    return parsed
        # try to dict() convert
        try:
            d = dict(result)
            if isinstance(d, dict) and d:
                return d
        except Exception:
            pass
        # parse str(result)
        parsed = try_extract_json_from_string(str(result))
        if parsed:
            return parsed
        # fallback wrappers
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return {"raw": result}
        return {"repr": repr(result)}
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}


def is_indian_movie(m: Dict[str, Any]) -> bool:
    """Heuristic: check _raw language/country or title tokens to mark Indian films."""
    if not isinstance(m, dict):
        return False
    raw = m.get("_raw") or m.get("raw") or {}
    if isinstance(raw, dict):
        lang = (raw.get("Language") or raw.get("language") or "").lower()
        country = (raw.get("Country") or raw.get("country") or "").lower()
        if "india" in country or "hindi" in lang:
            return True
    if isinstance(raw, str):
        s = raw.lower()
        if "india" in s or "hindi" in s:
            return True
    title = (m.get("Title") or m.get("title") or "").lower()
    for tok in ("raj", "bahubali", "gully", "raazi", "padma", "tanhaji", "drishyam", "singham"):
        if tok in title:
            return True
    return False


def normalize_movie_boxoffice(m: Dict[str, Any], usd_to_inr: float) -> Tuple[Optional[int], Optional[int]]:
    """Return (box_inr, budget_inr). Accepts multiple field namings."""
    def parse_money_str(s):
        if s is None:
            return None
        s = str(s)
        if s.upper() == "N/A":
            return None
        digits = re.sub(r"[^0-9]", "", s)
        if not digits:
            return None
        try:
            return int(digits)
        except Exception:
            return None

    box_inr = None
    budget_inr = None

    # common keys (case variations)
    for key in ("box_office_inr", "BoxOfficeINR", "box_office_inr"):
        if isinstance(m.get(key), (int, float)):
            box_inr = int(m.get(key))
            break
    if box_inr is None:
        raw_box = m.get("box_office") or m.get("BoxOffice") or m.get("BoxOfficeUSD") or m.get("boxOffice")
        parsed = parse_money_str(raw_box)
        if parsed:
            if is_indian_movie(m):
                box_inr = parsed
            else:
                box_inr = int(parsed * usd_to_inr)

    for key in ("budget_inr", "BudgetINR", "budget"):
        if isinstance(m.get(key), (int, float)):
            budget_inr = int(m.get(key))
            break
    if budget_inr is None:
        raw_budget = m.get("Budget") or m.get("budget")
        parsed_b = parse_money_str(raw_budget)
        if parsed_b:
            if is_indian_movie(m):
                budget_inr = parsed_b
            else:
                budget_inr = int(parsed_b * usd_to_inr)

    return box_inr, budget_inr


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Right Casting Choice AI", layout="wide")
st.title("🎬 Right Casting Choice AI — Candidate Pool")

with st.sidebar:
    st.header("Settings")
    n_similar = st.number_input("Number of similar movies", min_value=1, max_value=10, value=3)
    user_budget_ui = st.number_input("User budget (UI units - Cr for Bollywood, M for Hollywood)", min_value=0.0, value=100.0, step=0.5)
    usd_to_inr = float(st.number_input("USD→INR rate", min_value=10.0, max_value=200.0, value=83.0))
    industry = st.selectbox("Industry", options=["bollywood", "hollywood"], index=0)

# display units
if industry == "bollywood":
    currency_symbol = "₹"
    unit_label = "Cr"
    multiplier = 10_000_000
else:
    currency_symbol = "$"
    unit_label = "M"
    multiplier = 1_000_000

plot = st.text_area("Movie plot (short)", height=140, value="A righteous and fearless police officer takes on corruption and crime to restore justice in his city.")
run = st.button("Run Crew Flow")

if not run:
    st.info("Adjust settings and click 'Run Crew Flow' to fetch results from the crew.")
    st.stop()

st.info("Running crew.kickoff — this will execute the pipeline once and print results below.")
user_budget_raw = int(user_budget_ui * multiplier)

inputs = {
    "plot": plot,
    "n_similar": int(n_similar),
    "usd_to_inr": float(usd_to_inr),
    "user_budget_inr": int(user_budget_raw) if industry == "bollywood" else None,
    "user_budget_usd": int(user_budget_raw) if industry == "hollywood" else None,
    "industry": industry,
}

crew_output = safe_crew_kickoff(inputs)

# Try normalize if top-level is json-string
if isinstance(crew_output, str):
    parsed = try_extract_json_from_string(crew_output)
    crew_output = parsed or {"raw": crew_output}

# If 'tasks' exist and have raw JSON strings, try parsing them into 'output'
if isinstance(crew_output, dict) and "tasks" in crew_output:
    for t in crew_output.get("tasks", []):
        if not isinstance(t, dict):
            continue
        if t.get("output") is None:
            raw = t.get("raw_output") or t.get("raw") or t.get("result")
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                if parsed is not None:
                    t["output"] = parsed

# -------------------------
# Extract canonical pieces:
#   - characters (from extract_characters_task)
#   - per-role candidates (rank_candidates_task expected to return a JSON array per role)
#   - movies (from similar_movies_task)
# -------------------------
characters: List[Dict[str, Any]] = []
role_candidates: List[Dict[str, Any]] = []  # expected: list of {role, name_hint, candidates, movies}
movies_list: List[Dict[str, Any]] = []
recommended_pool: List[Dict[str, Any]] = []

if isinstance(crew_output, dict):
    # top-level characters
    chars_top = crew_output.get("extract_characters_task") or crew_output.get("characters") or crew_output.get("character_list")
    if isinstance(chars_top, list):
        characters = chars_top

    # top-level role candidates (maybe rank_candidates_task returned a list)
    rank_top = crew_output.get("rank_candidates_task") or crew_output.get("rank_candidates") or crew_output.get("recommendations") or crew_output.get("role_candidates")
    if isinstance(rank_top, list):
        role_candidates = rank_top
    elif isinstance(rank_top, dict):
        # maybe it's wrapped {roles: [...]} or similar
        role_candidates = rank_top.get("roles") or rank_top.get("per_role") or []

    # candidates / recommended_pool top-level
    recommended_pool = crew_output.get("recommended_pool") or crew_output.get("recommendations") or recommended_pool

    # movies top-level
    movies_list = crew_output.get("movies") or crew_output.get("movie_list") or crew_output.get("similar_movies") or []

    # search tasks for extract_characters_task and rank_candidates_task outputs if not found
    tasks = crew_output.get("tasks") or []
    if isinstance(tasks, list):
        for t in tasks:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").lower()
            out = t.get("output") or t.get("result") or t.get("raw_output") or t.get("raw")
            # characters
            if name in ("extract_characters_task", "character_extractor", "extract_characters") and isinstance(out, list):
                characters = out
            # role candidates (list-of-role dicts)
            if name in ("rank_candidates_task", "rank_candidates", "budget_ranker") and isinstance(out, list):
                role_candidates = out
            # movies
            if name in ("similar_movies_task", "similar_movies_and_omdb", "similar_movies") and isinstance(out, list):
                movies_list = out

# fallback heuristics: if role_candidates empty but a flat candidates list exists, try to build a simple mapping
flat_candidates = []
if isinstance(crew_output, dict):
    flat_candidates = crew_output.get("candidates") or crew_output.get("candidates_list") or []
    # guess movies_list from generic "movies" or nested structures
    if not movies_list:
        movies_list = crew_output.get("movies") or []

# If extract_characters_task returned a JSON-string somewhere, try to parse it
if not characters:
    # search recursively for lists that look like character profiles
    def find_char_lists(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict) and ("role" in obj[0] or "name_hint" in obj[0] or "age_range" in obj[0]):
                return obj
            for it in obj:
                res = find_char_lists(it)
                if res:
                    return res
        elif isinstance(obj, dict):
            for v in obj.values():
                res = find_char_lists(v)
                if res:
                    return res
        return None
    maybe_chars = find_char_lists(crew_output)
    if maybe_chars:
        characters = maybe_chars

# If still no character list, create a default Lead role for UI
if not characters:
    characters = [{"role": "Lead", "name_hint": "Lead", "age_range": "", "traits": []}]

# Normalize movies for display (priority scoring)
normalized_movies = []
for m in movies_list or []:
    if not isinstance(m, dict):
        continue
    poster = m.get("Poster") or m.get("poster") or m.get("poster_url") or ""
    indian = is_indian_movie(m)
    box_inr, budget_inr = normalize_movie_boxoffice(m, usd_to_inr)
    imdb_rating = 0.0
    try:
        imdb_rating = float(m.get("imdbRating") or m.get("imdb_rating") or 0)
    except Exception:
        imdb_rating = 0.0
    pref = 0.0
    if industry == "bollywood" and indian:
        pref += 2.0
    if industry == "hollywood" and not indian:
        pref += 2.0
    if poster:
        pref += 1.0
    pref += imdb_rating / 10.0
    normalized_movies.append({
        "raw": m,
        "Title": m.get("Title") or m.get("title") or "",
        "Year": m.get("Year") or m.get("year") or "",
        "Poster": poster,
        "imdbRating": imdb_rating,
        "BoxOfficeINR": box_inr,
        "BudgetINR": budget_inr,
        "is_indian": indian,
        "preference_score": pref,
        "imdbID": m.get("imdbID") or m.get("movie_id") or m.get("id"),
    })
normalized_movies = sorted(normalized_movies, key=lambda x: (-x["preference_score"], -float(x.get("imdbRating") or 0)))

# Build per-role actors mapping
# Role candidates can be: role_candidates (list of role dicts), or rank_candidates_task output array
actors_by_character: List[Dict[str, Any]] = []

# If role_candidates is a per-role JSON array (as per your tasks.yaml), use that structure directly
if isinstance(role_candidates, list) and role_candidates:
    # expected structure: [{"role":"Lead Male","name_hint":"Arjun","candidates":[...],"movies":[...]}]
    for rc in role_candidates:
        if not isinstance(rc, dict):
            continue
        role_name = rc.get("role") or rc.get("name_hint") or rc.get("role_name") or "Role"
        cand_list = rc.get("candidates") or rc.get("candidates_list") or rc.get("rows") or []
        # normalize candidate entries to include poster/imdb match if possible
        normalized_cands = []
        for a in cand_list:
            if not isinstance(a, dict):
                continue
            actor_name = a.get("name") or a.get("actor") or a.get("actor_name")
            score = a.get("score") or a.get("combined_score") or a.get("popularity_score") or a.get("raw_score")
            # try match poster/imdbRating via movie ids in a.movies or rc.movies
            poster = a.get("poster_url") or a.get("poster") or a.get("Poster") or ""
            imdb_rating = a.get("imdbRating") or a.get("imdb_rating")
            # if actor has listed movies, try to match a movie and attach its poster/imdb/bo
            if not poster:
                movie_ids = []
                if isinstance(a.get("movies"), list):
                    for mi in a["movies"]:
                        if isinstance(mi, dict):
                            movie_ids.append(mi.get("imdbID") or mi.get("movie_id") or mi.get("id"))
                        else:
                            movie_ids.append(mi)
                # also check rc.movies
                if not movie_ids and isinstance(rc.get("movies"), list):
                    for mi in rc["movies"]:
                        if isinstance(mi, dict):
                            movie_ids.append(mi.get("imdbID") or mi.get("movie_id") or mi.get("id"))
                # find first matching movie in normalized_movies
                for m in normalized_movies:
                    if m.get("imdbID") and str(m.get("imdbID")) in [str(x) for x in movie_ids if x]:
                        poster = poster or m.get("Poster")
                        imdb_rating = imdb_rating or m.get("imdbRating")
                        break
            normalized_cands.append({
                "name": actor_name,
                "score": score,
                "movies": a.get("movies") or [],
                "poster": poster,
                "imdbRating": imdb_rating,
                "raw": a,
            })
        actors_by_character.append({"role": role_name, "traits": rc.get("traits") or [], "actors": normalized_cands})
else:
    # fallback: use flat candidates_list or recommended_pool to assign to roles heuristically
    source = flat_candidates or (crew_output.get("recommended_pool") if isinstance(crew_output, dict) else [])
    for ch in characters:
        role_name = ch.get("role") or ch.get("name") or "Role"
        cand_entries = []
        for a in source:
            if not isinstance(a, dict):
                continue
            # heuristics: if actor has role_matches/role field, prefer those matching this role
            role_matches = a.get("role_matches") or a.get("roles") or []
            # include actor if role_matches contains role_name or include all (best-effort)
            include = False
            if isinstance(role_matches, list) and role_matches:
                include = any(role_name.lower() in str(r).lower() for r in role_matches)
            else:
                include = True  # best-effort include
            if include:
                name = a.get("name") or a.get("actor")
                poster = a.get("poster_url") or a.get("poster") or ""
                # try to match poster from movies
                movie_ids = []
                if isinstance(a.get("movies"), list):
                    for mi in a["movies"]:
                        if isinstance(mi, dict):
                            movie_ids.append(mi.get("imdbID") or mi.get("movie_id") or mi.get("id"))
                for m in normalized_movies:
                    if m.get("imdbID") and str(m.get("imdbID")) in [str(x) for x in movie_ids if x]:
                        poster = poster or m.get("Poster")
                        break
                cand_entries.append({
                    "name": name,
                    "score": a.get("score") or a.get("raw_score") or a.get("popularity_score"),
                    "movies": a.get("movies") or [],
                    "poster": poster,
                    "raw": a,
                })
        actors_by_character.append({"role": role_name, "traits": ch.get("traits") or [], "actors": cand_entries})

# ensure actors_by_character has entries
if not actors_by_character:
    # build a simple single role using recommended_pool or flat candidates
    fallback_actors = []
    src = recommended_pool or flat_candidates or []
    for a in src:
        if isinstance(a, dict):
            fallback_actors.append({
                "name": a.get("name") or a.get("actor"),
                "score": a.get("score") or a.get("raw_score"),
                "movies": a.get("movies") or [],
                "poster": a.get("poster") or a.get("Poster") or ""
            })
    actors_by_character = [{"role": "Lead", "traits": [], "actors": fallback_actors}]

# -------------------------
# UI: Tabs -> Candidate Pool, Similar Movies
# -------------------------
tab1, tab2 = st.tabs(["📋 Candidate Pool", "🎞️ Similar Movies"])

with tab1:
    st.header(f"Audition List ({industry.capitalize()})")
    for ch in actors_by_character:
        role = ch.get("role") or "Role"
        name_hint = ch.get("name_hint") or ""
        traits = ch.get("traits") or []
        header = f"🎭 Role: {role}"
        if name_hint:
            header += f" — {name_hint}"
        with st.expander(header, expanded=False):
            if traits:
                st.markdown(f"**Traits:** {', '.join(traits)}")
            actors = ch.get("actors") or []
            if not actors:
                st.write("No actors found for this role.")
                continue
            # build dataframe for display
            rows = []
            for a in actors:
                # attempt to fill salary/box/rating/versatility/risk if present in raw
                raw = a.get("raw") or {}
                salary = raw.get("implied_actor_fee_estimate") or raw.get("salary") or raw.get("fee") or None
                bo = None
                # if actor has movie entries, use first movie's BoxOffice if we have normalized movie list
                movie_ref = None
                if a.get("movies"):
                    first = a["movies"][0]
                    if isinstance(first, dict):
                        movie_ref = first.get("imdbID") or first.get("movie_id") or first.get("id")
                imdb_rating = a.get("imdbRating") or raw.get("imdbRating") or raw.get("rating") or None
                versatility = raw.get("versatility") or raw.get("versatility_score") or raw.get("versatility_index") or None
                risk = raw.get("risk") or raw.get("risk_factor") or raw.get("risk_score") or None
                poster = a.get("poster") or ""
                # try to find BO if movie_ref exists
                if movie_ref:
                    for m in normalized_movies:
                        if m.get("imdbID") and str(m.get("imdbID")) == str(movie_ref):
                            bo = m.get("BoxOfficeINR") or m.get("BoxOfficeUSD")
                            imdb_rating = imdb_rating or m.get("imdbRating")
                            poster = poster or m.get("Poster")
                            break
                # convert salary/bo into UI units where applicable
                salary_display = ""
                bo_display = ""
                try:
                    if salary:
                        salary_display = f"{currency_symbol}{salary / multiplier:,.2f} {unit_label}"
                except Exception:
                    salary_display = salary
                try:
                    if bo:
                        bo_display = f"{currency_symbol}{(bo / multiplier):,.2f} {unit_label}"
                except Exception:
                    bo_display = bo
                rows.append({
                    "Actor Name": a.get("name") or "Unknown",
                    "Salary (UI)": salary_display,
                    "Proj. BO (UI)": bo_display,
                    "Rating": float(imdb_rating) if imdb_rating not in (None, "") else None,
                    "versatility": versatility,
                    "Risk Factor": risk,
                    "Poster": poster
                })
            df = pd.DataFrame(rows)
            # display dataframe (without Poster column)
            display_df = df.drop(columns=["Poster"], errors="ignore")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # show posters (max 6)
            posters = [r for r in df["Poster"].tolist() if r]
            if posters:
                cols = st.columns(min(len(posters), 6))
                for i, p in enumerate(posters[:6]):
                    with cols[i]:
                        st.image(p, use_column_width=True)

with tab2:
    st.header("Similar Movies")
    if normalized_movies:
        # Show top-n posters
        posters = [m["Poster"] for m in normalized_movies if m.get("Poster")]
        if posters:
            cols = st.columns(min(len(posters[:n_similar]), n_similar))
            for i, p in enumerate(posters[:n_similar]):
                with cols[i]:
                    st.image(p, use_column_width=True)
        # show compact movie table
        rows = []
        for m in normalized_movies:
            rows.append({
                "Title": m["Title"],
                "Year": m["Year"],
                "IMDB": m["imdbRating"],
                f"BoxOffice ({'INR' if industry=='bollywood' else 'USD'})": (m["BoxOfficeINR"] if industry == "bollywood" else m["BoxOfficeUSD"]) or ""
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No similar movie metadata found to display.")
# Show raw for debugging
st.subheader("Raw Crew Output (abridged)")
st.json(crew_output if isinstance(crew_output, dict) else {"raw": crew_output})
st.success("Crew run complete — see Candidate Pool and Similar Movies.")
