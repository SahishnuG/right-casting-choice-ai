# app_all_fixes.py — consolidated fixes: strict extractor/ranker candidates, fixed selection, cleaner parsing, optional match score
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import time
import os

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

from right_casting_choice_ai.crew import RightCastingChoiceAi

# ---- helpers ----

def try_extract_json_from_string(s: Optional[str]) -> Optional[Any]:
    if s is None:
        return None
    text = str(s)
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
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    for candidate in (cleaned, cleaned.replace("'", '"')):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    arrays = list(re.finditer(r"\[[\s\S]*?\]", cleaned))
    for m in reversed(arrays):
        cand = m.group(0)
        for candidate in (cand, cand.replace("'", '"')):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    objects = list(re.finditer(r"\{[\s\S]*?\}", cleaned))
    for m in reversed(objects):
        cand = m.group(0)
        for candidate in (cand, cand.replace("'", '"')):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None


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
                tasks_out = result.tasks_output
                last_error = None
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                is_rate_limit = ("RESOURCE_EXHAUSTED" in msg or "RateLimit" in msg or "quota" in msg.lower())
                if not is_rate_limit or attempt >= attempts - 1:
                    break
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
        # normalize result
        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict) and d:
                    return d, tasks_out
            except Exception:
                pass
        if hasattr(result, "raw"):
            raw = result.raw
            if isinstance(raw, dict):
                return raw, tasks_out
            if isinstance(raw, list):
                return {"raw": raw}, tasks_out
            if isinstance(raw, str):
                parsed = try_extract_json_from_string(raw)
                return {"raw": parsed} if parsed is not None else {"raw": raw}, tasks_out
        try:
            d = dict(result)
            if isinstance(d, dict):
                return d, tasks_out
        except Exception:
            pass
        s = str(result)
        parsed = try_extract_json_from_string(s)
        if parsed is not None:
            return {"raw": parsed}, tasks_out
        if last_error is not None:
            return {"error": f"Crew kickoff failed: {last_error}"}, None
        return {"raw": str(result)}, tasks_out
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}"}, None


def pick_first(obj: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return v
    return default


def parse_money_str_to_int(s: Optional[str], usd_to_inr: float = 83.0) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.upper() == "N/A":
        return None
    is_usd = bool(re.search(r"^\s*\$", s) or re.search(r"\$[0-9,]", s))
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    try:
        val = int(digits)
        if is_usd:
            return int(val * usd_to_inr)
        return int(val)
    except Exception:
        return None


def normalize_movie_boxoffice(m: Dict[str, Any], usd_to_inr: float) -> Tuple[Optional[int], Optional[int]]:
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


def extract_roles_and_movies_from_crew(tasks, crew_out: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    roles: List[Dict[str, Any]] = []
    movies: List[Dict[str, Any]] = []
    if not isinstance(crew_out, dict):
        return roles, movies
    raw = crew_out.get("raw")
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and ("role" in raw[0] or "candidates" in raw[0] or "name_hint" in raw[0]):
            roles = raw
        else:
            for v in raw:
                if isinstance(v, dict):
                    if "role" in v or "candidates" in v or "name_hint" in v:
                        roles.append(v)
                    if "imdbID" in v or "Title" in v or "Poster" in v:
                        movies.append(v)
    elif isinstance(raw, dict):
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
    if not roles:
        if isinstance(crew_out.get("candidates"), list) or isinstance(crew_out.get("recommended_pool"), list):
            roles = [{"role": crew_out.get("role") or "Lead", "candidates": crew_out.get("candidates") or crew_out.get("recommended_pool"), "movies": crew_out.get("movies") or []}]
        elif isinstance(crew_out.get("movies"), list):
            movies = crew_out.get("movies")
    # Consider per-task outputs if provided under 'tasks_output'
    movies = try_extract_json_from_string(tasks[1]) 
    return roles, movies


def parse_fee_to_int(value: Any) -> Optional[int]:
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
    if len(nums) >= 2:
        first, last = nums[0], nums[-1]
        return int((first + last) / 2)
    return None


def compute_candidate_stats(candidate: Dict[str, Any], movies_for_role: List[Dict[str, Any]], usd_to_inr: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["name"] = pick_first(candidate, ["name", "actor", "actor_name", "title"], "Unknown")
    c_imdb = pick_first(candidate, ["imdb_rating", "average_imdb_rating", "avg_imdb", "imdb"], None)
    out["avg_imdb"] = float(c_imdb) if (c_imdb is not None and str(c_imdb).replace('.', '').isdigit()) else None
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
    if not matched:
        nm = out["name"]
        if nm and movies_for_role:
            for m in movies_for_role:
                actors_str = (m.get("Actors") or m.get("actors") or "")
                if isinstance(actors_str, str) and nm.lower() in actors_str.lower():
                    matched.append(m)
    imdb_vals = []
    box_vals = []
    for mm in matched:
        try:
            mr = mm.get("imdbRating") or mm.get("imdb_rating") or mm.get("imdb")
            if mr is not None:
                imdb_vals.append(float(mr))
        except Exception:
            pass
        box_inr, _ = normalize_movie_boxoffice(mm, usd_to_inr)
        if isinstance(box_inr, (int, float)):
            box_vals.append(int(box_inr))
    if out["avg_imdb"] is None and imdb_vals:
        out["avg_imdb"] = sum(imdb_vals) / len(imdb_vals)
    out["avg_box_inr"] = int(sum(box_vals) / len(box_vals)) if box_vals else None
    fee_raw = pick_first(candidate, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None)
    out["fee_inr"] = parse_fee_to_int(fee_raw)
    return out


# ---- Streamlit UI ----
st.set_page_config(page_title="Right Casting Choice AI — app1", layout="wide")
st.title("🎬 Right Casting Choice AI")

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

with st.spinner("Running crew pipeline..."):
    crew_output, tasks_output = safe_crew_kickoff(inputs)

if isinstance(crew_output, dict) and crew_output.get("error"):
    st.error("Gemini rate limit/quota hit. Please wait and retry.")
    st.caption(str(crew_output.get("error")))
    st.stop()

roles_list, movies_list = extract_roles_and_movies_from_crew(tasks_output, crew_output)

# Detect characters for summary only (no retrying)

def detect_characters_from_tasks(tasks: Dict[str, Any]) -> List[Dict[str, Any]]:
    chars: List[Dict[str, Any]] = []
    chars = try_extract_json_from_string(tasks[0])
    print(chars)
    return chars

initial_chars = detect_characters_from_tasks(tasks_output)

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

# Display detected characters (summary only)
if initial_chars:
    st.subheader("Detected Characters")
    det_rows = []
    for c in initial_chars:
        nm = pick_first(c, ["name_hint", "role", "name"], None) or "Unknown"
        gen = pick_first(c, ["gender", "sex"], "")
        age = pick_first(c, ["age_range", "estimated_age_at_peak_performance_for_role"], "")
        traits = c.get("traits") or c.get("notes") or []
        traits_str = ", ".join(map(str, traits)) if isinstance(traits, list) else str(traits)
        det_rows.append({"Character": nm, "Gender": gen, "Age Range": age, "Traits": traits_str})
    try:
        st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
    except Exception:
        st.write(det_rows)
    st.caption(f"Detected {len(initial_chars)} character(s) from the extractor.")

# Tabs
tab_candidates, tab_movies = st.tabs(["📋 Candidate Pool", "🎞 Similar Movies & Posters"])

with tab_candidates:
    st.header(f"Audition List ({industry.capitalize()})")

    def normalize_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
        nc: Dict[str, Any] = {}
        nc["name"] = pick_first(c, ["name", "actor", "actor_name", "title"], "Unknown")
        ir = pick_first(c, ["imdb_rating", "average_imdb_rating", "avg_imdb", "imdb"], None)
        try:
            nc["imdb_rating"] = float(ir) if ir is not None else None
        except Exception:
            nc["imdb_rating"] = None
        abo = pick_first(c, ["average_box_office_inr", "avg_box_office_inr", "avg_box_inr", "box_office_inr"], None)
        try:
            nc["average_box_office_inr"] = int(abo) if abo is not None else None
        except Exception:
            nc["average_box_office_inr"] = None
        fee_raw = pick_first(c, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None)
        nc["implied_actor_fee_estimate"] = parse_fee_to_int(fee_raw)
        nc["fee_source"] = pick_first(c, ["fee_source", "notes", "note", "explanation"], "")
        sc = pick_first(c, ["score", "combined_score", "raw_score"], None)
        try:
            nc["score"] = float(sc) if sc is not None else None
        except Exception:
            nc["score"] = None
        return nc

    def normalize_role_block(rb: Dict[str, Any]) -> Dict[str, Any]:
        nr: Dict[str, Any] = {}
        nr["role"] = rb.get("role") or "Role"
        nr["name_hint"] = pick_first(rb, ["name_hint", "nameHint"], "")
        nr["age_range"] = pick_first(rb, ["age_range", "estimated_age_at_peak_performance_for_role"], "")
        nr["gender"] = rb.get("gender") or ""
        nr["traits"] = rb.get("traits") or rb.get("notes") or []
        mlist = rb.get("movies") or movies_list or []
        nr["movies"] = mlist if isinstance(mlist, list) else []
        cand_list = rb.get("candidates") or []
        if not isinstance(cand_list, list):
            cand_list = []
        nr["candidates"] = [normalize_candidate(c) for c in cand_list if isinstance(c, dict)]
        if isinstance(rb.get("recommended_pool"), list) and rb.get("recommended_pool"):
            rp = [normalize_candidate(c) for c in rb["recommended_pool"] if isinstance(c, dict)]
            nr["recommended_pool"] = rp
        else:
            sorted_pool = sorted(nr["candidates"], key=lambda x: (x.get("score") or 0.0, - (x.get("implied_actor_fee_estimate") or 0)), reverse=True)
            nr["recommended_pool"] = sorted_pool[:6]
        return nr

    if not roles_list:
        st.info("No role/candidate data found in crew output. See raw output below.")
    else:
        roles_list = [normalize_role_block(r) for r in roles_list if isinstance(r, dict)]
        grand_total_fee_inr = 0
        grand_roles_with_selected = 0
        selected_actor_names: set[str] = set()

        def _low_trust_fee_source(src: str) -> bool:
            if not src:
                return True
            s = src.lower()
            if ("networth" in s) or ("net worth" in s) or ("celebritynetworth" in s):
                return True
            if ("various industry reports" in s) or ("media reports" in s) or ("sources say" in s):
                return True
            if "http" not in s:
                return True
            return False

        for role_block in roles_list:
            role_name = role_block.get("role") or "Role"
            name_hint = role_block.get("name_hint") or role_block.get("nameHint") or ""
            age_hint = role_block.get("age_range") or role_block.get("estimated_age_at_peak_performance_for_role") or ""
            gender_hint = role_block.get("gender") or ""
            traits = role_block.get("traits") or role_block.get("notes") or []

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
                candidates = role_block.get("candidates") or []  # no synthetic fallback

                rows = []
                selected_candidate_fee_inr: Optional[int] = None
                selected_candidate_name: Optional[str] = None

                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    stats = compute_candidate_stats(c, movies_for_role, usd_to_inr)
                    name = stats.get("name")
                    avg_imdb = stats.get("avg_imdb")
                    avg_bo_inr = stats.get("avg_box_inr")
                    score = pick_first(c, ["score", "combined_score", "raw_score"], None)
                    fee_inr = stats.get("fee_inr") if "fee_inr" in stats else None
                    if fee_inr is None:
                        fee_inr = parse_fee_to_int(pick_first(c, ["implied_actor_fee_estimate", "implied_fee", "fee", "estimated_fee", "actor_fee_inr"], None))
                    src_txt = pick_first(c, ["fee_source", "notes", "note", "explanation"], "") or ""
                    low_trust = _low_trust_fee_source(src_txt)
                    fee_inr_for_selection = None if low_trust else fee_inr

                    fee_display = ""
                    if isinstance(fee_inr, int):
                        if industry == "hollywood":
                            fee_usd = fee_inr / usd_to_inr
                            fee_display = f"${(fee_usd / 1_000_000):,.2f} M"
                        else:
                            fee_display = f"₹{(fee_inr / 10_000_000):,.2f} Cr"

                    # UI selection: take the first candidate from crew output per role, skip duplicates across roles
                    if selected_candidate_name is None:
                        if isinstance(name, str) and name.strip() and name.strip().lower() in {n.lower() for n in selected_actor_names}:
                            pass
                        else:
                            selected_candidate_name = name
                            selected_candidate_fee_inr = fee_inr

                    if isinstance(avg_bo_inr, (int, float)):
                        if industry == "hollywood":
                            avg_bo_display = f"${((avg_bo_inr / usd_to_inr) / 1_000_000):,.2f} M"
                        else:
                            avg_bo_display = f"₹{(avg_bo_inr / 10_000_000):,.2f} Cr"
                    else:
                        avg_bo_display = ""

                    note_src = src_txt
                    if low_trust:
                        note_src = f"{src_txt} [flagged: low-trust source ignored for fee]"

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
                        grand_total_fee_inr += int(selected_candidate_fee_inr)
                    else:
                        st.markdown(f"**Selected Top Candidate:** {selected_candidate_name} — Fee: ₹{selected_candidate_fee_inr/10_000_000:,.2f} Cr")
                        grand_total_fee_inr += selected_candidate_fee_inr
                    if isinstance(selected_candidate_name, str) and selected_candidate_name.strip():
                        selected_actor_names.add(selected_candidate_name.strip())
                else:
                    st.markdown("**Selected Top Candidate:** No fee data available; cost omitted.")
                grand_roles_with_selected += 1

        st.markdown("---")
        if grand_roles_with_selected:
            if industry == "hollywood":
                total_usd = (grand_total_fee_inr / usd_to_inr)
                st.markdown(f"### Total cost (top candidate per role): **${total_usd/1_000_000:,.2f} M**")
                st.markdown(f"### Your budget: **${user_budget_raw/1_000_000:,.2f} M**")
                remaining_usd = user_budget_raw - total_usd
                rem_display = f"${remaining_usd/1_000_000:,.2f} M"
                if remaining_usd < 0:
                    st.error(f"Budget exceeded. Remaining: {rem_display}")
                else:
                    st.success(f"Remaining budget after counted fees: {rem_display}")
            else:
                st.markdown(f"### Total cost (top candidate per role): **₹{grand_total_fee_inr/10_000_000:,.2f} Cr**")
                st.markdown(f"### Your budget: **₹{user_budget_raw/10_000_000:,.2f} Cr**")
                remaining = user_budget_raw - grand_total_fee_inr
                rem_display = f"₹{remaining/10_000_000:,.2f} Cr"
                if remaining < 0:
                    st.error(f"Budget exceeded. Remaining: {rem_display}")
                else:
                    st.success(f"Remaining budget after counted fees: {rem_display}")
            st.caption("Note: UI selects the first candidate returned by the crew for each role. Duplicate actors across roles are prevented.")
        else:
            st.info("No actor fee fields were parsed; total cost unavailable. Ensure your budget_ranker returns numeric implied_actor_fee_estimate in INR.")

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
            print("Poster URL:", poster)
            title = pick_first(m, ["Title", "title", "name"]) or ""
            year = pick_first(m, ["Year", "year"]) or ""
            imdb = pick_first(m, ["imdbRating", "imdb_rating", "imdb"]) or ""
            box_inr, _ = normalize_movie_boxoffice(m, usd_to_inr)
            match_score = m.get("match_score")
            if poster and poster not in seen_posters:
                posters.append({"poster": poster, "title": title})
                seen_posters.add(poster)
            row = {
                "Title": title,
                "Year": year,
                "IMDB": imdb,
                "BoxOffice (INR)" if industry == "bollywood" else "BoxOffice (USD)": (
                    box_inr if industry == "bollywood" else (int(box_inr / usd_to_inr) if box_inr else None)
                )
            }
            if isinstance(match_score, (int, float)):
                row["Match Score"] = round(float(match_score), 3)
            rows.append(row)
        if posters:
            cols = st.columns(min(len(posters), n_similar))
            for i, p in enumerate(posters[:n_similar]):
                with cols[i]:
                    st.image(p["poster"], use_container_width=True)
                    st.caption(p["title"])
        try:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception:
            st.write(rows)

# Print each task result if tasks_output present
if isinstance(tasks_output, list) and tasks_output:
    st.subheader("Task Outputs")
    for i, t in enumerate(tasks_output, start=1):
        st.markdown(f"**Task {i}:**")
        if isinstance(t, str):
            parsed = try_extract_json_from_string(t)
            st.json(parsed if parsed is not None else t)
        else:
            st.json(t)

st.success("Done — candidate pool and similar movies displayed.")
