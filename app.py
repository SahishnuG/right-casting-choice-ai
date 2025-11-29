"""
Right Casting Choice AI - Streamlit App (simplified)

This simplified version does exactly what you asked:
- Runs a single `crew.kickoff` to execute the pipeline.
- Formats and prints the Crew output cleanly in Streamlit.
- Displays `n_similar` posters (from OMDb tool `Poster` field) in the app UI.
- Keeps error handling minimal but robust: if Crew or tools fail, prints the fallback outputs.
- Accepts user budget in CRORE (INR) and converts to INR before passing to the crew.

Usage:
    pip install streamlit requests pandas numpy crewai-tools google-generativeai python-dotenv
    setx GEMINI_API_KEY "<your-gemini-key>"  # or setx GOOGLE_API_KEY
    setx SERPER_API_KEY "<your-serper-key>"
    setx OMDB_API_KEY "<your-omdb-key>"
    streamlit run right-casting-choice-ai_streamlit.py

"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env early and standardize key env vars for LLM/tools
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

# Ensure src/ is importable
sys.path.append(str((Path(__file__).parent / "src").resolve()))

# Crew and tools
from right_casting_choice_ai.crew import RightCastingChoiceAi
from right_casting_choice_ai.tools.omdb import OmdbTool

# Minimal Gemini fallback (used only for local character extraction if needed)
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    genai = None
    _HAS_GEMINI = False


def convert_crore_to_inr(crore: float) -> int:
    return int(crore * 10_000_000)


def safe_crew_kickoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run single crew.kickoff and return a dict (best effort)."""
    try:
        crew = RightCastingChoiceAi().crew()
        result = crew.kickoff(inputs=inputs)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if hasattr(result, "raw"):
            return result.raw
        try:
            return dict(result)
        except Exception:
            return {"raw_result": str(result)}
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}


st.set_page_config(page_title="Right Casting Choice AI (simple)", layout="wide")
st.title("🎬 Right Casting Choice AI — Simplified")

with st.sidebar:
    st.header("Settings")
    n_similar = st.number_input("Number of similar movies", min_value=1, max_value=10, value=3)
    user_budget_crore = st.number_input("User budget (INR crores)", min_value=0.0, value=10.0, step=0.5)
    usd_to_inr = float(st.number_input("USD→INR rate", min_value=10.0, max_value=200.0, value=83.0))
    industry = st.selectbox("Industry", options=["hollywood", "bollywood"], index=0)

plot = st.text_area("Movie plot (<2000 characters)", height=180, value="A righteous and fearless police officer takes on corruption and crime to restore justice in his city.")
run_btn = st.button("Run Crew Flow")

if run_btn:
    st.info("Running crew.kickoff — this will execute the pipeline once and print results below.")
    user_budget_inr = convert_crore_to_inr(user_budget_crore)

    inputs = {
        "plot": plot,
        "n_similar": int(n_similar),
        "usd_to_inr": float(usd_to_inr),
        "user_budget_inr": int(user_budget_inr),
        "industry": industry,
    }

    # Try to capture richer outputs from Crew when available
    crew_output = safe_crew_kickoff(inputs)
    print(crew_output)  # Debugging
    # If empty, inspect common attributes on CrewOutput
    if isinstance(crew_output, dict) and not crew_output:
        try:
            crew = RightCastingChoiceAi().crew()
            result_obj = crew.kickoff(inputs=inputs)
            # Try known attributes on CrewOutput to build a dict
            enriched = {}
            if hasattr(result_obj, "to_dict"):
                enriched = result_obj.to_dict() or {}
            elif hasattr(result_obj, "raw"):
                enriched = result_obj.raw or {}
            # tasks_output: list of per-task objects
            if not enriched and hasattr(result_obj, "tasks_output"):
                try:
                    tasks_out = []
                    for t in getattr(result_obj, "tasks_output", []):
                        # Try to read common fields
                        name = getattr(t, "name", None) or getattr(t, "task_name", None)
                        output = getattr(t, "output", None) or getattr(t, "result", None)
                        raw_output = getattr(t, "raw_output", None)
                        tasks_out.append({
                            "name": name,
                            "output": output,
                            "raw_output": raw_output,
                        })
                    enriched = {"tasks": tasks_out}
                except Exception:
                    pass
            # Fallback to repr
            if not enriched:
                enriched = {"repr": repr(result_obj)}
            crew_output = enriched
        except Exception:
            pass

    st.subheader("Raw Crew Output (JSON)")
    st.json(crew_output)
    # Debug keys view to help diagnose structure
    try:
        if isinstance(crew_output, dict):
            st.caption(f"Top-level keys: {list(crew_output.keys())}")
    except Exception:
        pass

    # Try to locate the OMDb/Similar movies results and display posters
    posters: List[str] = []

    # Common places Crew output may store similar movies
    candidates = []
    if isinstance(crew_output, dict):
        # pattern 1: top-level similar_movies_task
        maybe = crew_output.get("similar_movies_task") or crew_output.get("similar_movies")
        if isinstance(maybe, list):
            candidates = maybe
        else:
            # pattern 2: tasks list
            tasks = crew_output.get("tasks") or []
            if isinstance(tasks, list):
                for t in tasks:
                    if isinstance(t, dict) and t.get("name") in ("similar_movies_task", "similar_movies_and_omdb", "similar_movies"):
                        out = t.get("output") or t.get("result") or t.get("raw_output")
                        if isinstance(out, list):
                            candidates = out
                            break
            # pattern 3: results map
            if not candidates:
                results_map = crew_output.get("results") or {}
                maybe = results_map.get("similar_movies_task") or results_map.get("similar_movies")
                if isinstance(maybe, list):
                    candidates = maybe

    # If still empty, attempt to harvest any list of dicts with Title/Poster fields
    if not candidates:
        # search recursively for movie-like lists
        def find_movie_lists(obj):
            found = []
            if isinstance(obj, list):
                # check if list elements are dicts with Title
                if all(isinstance(i, dict) and ("Title" in i or "Poster" in i) for i in obj):
                    return obj
                for item in obj:
                    res = find_movie_lists(item)
                    if res:
                        return res
            elif isinstance(obj, dict):
                for v in obj.values():
                    res = find_movie_lists(v)
                    if res:
                        return res
            return None

        maybe = find_movie_lists(crew_output)
        if maybe:
            candidates = maybe

    # If bollywood, filter candidates/movies heuristically for Indian context
    if industry == "bollywood":
        filtered = []
        for item in candidates:
            if isinstance(item, dict):
                raw = item.get("_raw") or {}
                lang = (raw.get("Language") or "").lower()
                country = (raw.get("Country") or "").lower()
                if ("india" in country) or ("hindi" in lang):
                    filtered.append(item)
        if filtered:
            candidates = filtered

    # Collect poster URLs up to n_similar
    for item in candidates:
        if isinstance(item, dict):
            poster = item.get("Poster") or item.get("poster") or item.get("PosterUrl")
            if poster:
                posters.append(poster)
        if len(posters) >= n_similar:
            break

    # Display posters
    st.subheader(f"Top {n_similar} Posters from OMDb results")
    if posters:
        cols = st.columns(min(len(posters), n_similar))
        for idx, url in enumerate(posters[:n_similar]):
            c = cols[idx % len(cols)]
            with c:
                st.image(url, use_column_width=True)
    else:
        st.info("No posters found in Crew output. Ensure OMDb tool returned 'Poster' links in its output.")

    # Additionally, format and print actor ranking if available
    st.subheader("Actor Candidates / Rankings (if present)")
    ranking = None
    if isinstance(crew_output, dict):
        ranking = crew_output.get("rank_candidates_task") or crew_output.get("rank_candidates") or crew_output.get("ranking")
        if not ranking:
            tasks = crew_output.get("tasks") or []
            for t in tasks:
                if isinstance(t, dict) and t.get("name") in ("rank_candidates_task", "rank_candidates", "budget_ranker"):
                    ranking = t.get("output") or t.get("result") or t.get("raw_output")
                    break

    if isinstance(ranking, dict):
        candidates = ranking.get("candidates") or ranking.get("candidates_list") or ranking.get("rows")
        if isinstance(candidates, list):
            df = pd.DataFrame(candidates)
            st.dataframe(df)
            # Build per-role candidate lists
            st.subheader("Suggested Actors per Role")
            # Characters may come from crew output; try multiple locations
            characters: List[Dict[str, Any]] = []
            if isinstance(crew_output, dict):
                characters = crew_output.get("extract_characters_task") or crew_output.get("characters") or []
                if not isinstance(characters, list):
                    characters = []
            # If characters empty, create a single generic role
            if not characters:
                characters = [{"role": "Lead", "traits": ["heroic", "fearless"], "age_range": "25-45"}]
            # Use top-N actors as suggestions per role (simple heuristic)
            top_names: List[str] = []
            # candidates items may be dicts with 'name' field
            for row in candidates:
                name = row.get("name") or row.get("actor")
                if name:
                    top_names.append(name)
            top_names = top_names[:6] if top_names else []
            for ch in characters:
                role = ch.get("role") or "Role"
                st.markdown(f"**{role}**")
                st.write(", ".join(top_names) if top_names else "No candidates found")
        else:
            st.write(ranking)
    else:
        st.info("No actor ranking found in Crew output.")

    st.success("Crew run complete. See above for results.")
