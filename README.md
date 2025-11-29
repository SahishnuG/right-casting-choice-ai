# Right Casting Choice AI

Right Casting Choice AI is a three-agent casting workflow built on [crewAI](https://crewai.com). It analyzes a movie plot, finds similar films, enriches them with OMDb data, and ranks actors against budget constraints — producing candidate lists per role.

Key features:
- Character extraction via Gemini (JSON-only output)
- Similar movies found with Serper and enriched with OMDb (converted to INR)
- Budget-aware actor ranking and per-role suggestions
- Industry selection (Hollywood or Bollywood) with Bollywood-specific filtering

This project uses Gemini exclusively; there are no OpenAI calls. Environment variables are read from `.env`.

---

## Quick Start

Requirements:
- Python `>=3.10,<3.14`

Install dependencies:
```powershell
pip install -e .
```

Create a `.env` file in the project root:
```env
# Gemini
GEMINI_API_KEY=your_gemini_key   # or use GOOGLE_API_KEY

# Serper
SERPER_API_KEY=your_serper_key   # or SERPERDEV_API_KEY

# OMDb
OMDB_API_KEY=your_omdb_key

# Optional
USD_TO_INR_RATE=83.0
```

Run the Streamlit app:
```powershell
streamlit run app.py
```
In the UI:
- Enter the plot
- Choose `Industry` (Hollywood or Bollywood)
- Set `n_similar`, budget (crores for Bollywood or USD for Hollywood), and USD→INR rate

Run the CLI sample:
```powershell
python main.py
```
It prints summary counts, per-role suggestions, and a normalized JSON result.

---

## How It Works

Pipeline (sequential crew):
1. Character Extractor (`character_extractor`)
	 - LLM: `gemini/gemini-2.5-flash`
	 - Task: `extract_characters_task`
	 - Output: JSON array of 2–6 character profiles with: `role`, `name_hint`, `age_range`, `traits`, `gender`

2. Similar Movies + OMDb (`similar_movies_and_omdb`)
	 - Tooling: `SerperDevTool`, custom `OmdbTool`
	 - Task: `similar_movies_task`
	 - Output per movie: `{Title, Year, Actors, Poster, BoxOfficeINR, BudgetINR, imdbRating, imdbID, _raw}`
	 - Bollywood filter (UI): keeps movies whose OMDb `_raw.Language` contains “Hindi” or `_raw.Country` contains “India”

3. Budget Ranker (`budget_ranker`)
	 - Task: `rank_candidates_task`
	 - Expected output (tasks.yaml): per-role objects like:
		 ```json
		 [
			 {"candidates": [...], "role": "Lead Male", "name_hint": "Arjun", "movies": [...]},
			 {"candidates": [...], "role": "Lead Female", "name_hint": "Meera", "movies": [...]}
		 ]
		 ```
	 - The app also builds per-role suggestions by pairing the top-ranked actors with extracted roles if the task returns a flat candidate list.

Data flow:
- `app.py` orchestrates the crew run, normalizes the `CrewOutput`, and displays:
	- Candidate pool (actors table)
	- Suggested actors per role
	- Similar movies & posters
	- Raw parsed JSON (with top-level keys to aid debugging)

---

## Configuration & Code Layout

- `src/right_casting_choice_ai/crew.py`
	- Declares the three agents and tasks
	- Attaches Serper + OMDb tools to the second agent
	- Uses `gemini/gemini-2.5-flash` for all agents

- `src/right_casting_choice_ai/config/agents.yaml`
	- Agent roles, goals, backstories
	- LLM model string: `gemini/gemini-2.5-flash`

- `src/right_casting_choice_ai/config/tasks.yaml`
	- Task descriptions & expected outputs
	- Includes `industry` and `usd_to_inr` in prompts to guide tool usage

- `src/right_casting_choice_ai/tools/omdb.py`
	- `OmdbTool` (CrewAI BaseTool)
	- Queries OMDb and returns a JSON string with normalized fields
	- Parses USD BoxOffice/Budget and converts to INR (`USD_TO_INR_RATE`)

- `app.py`
	- Streamlit UI with two tabs: “Candidate Pool” and “Similar Movies & Posters”
	- Normalizes `CrewOutput` across versions (`to_dict`, `raw`, `results`, `tasks_output`)
	- Industry dropdown applies Bollywood filtering before poster rendering

- `main.py`
	- CLI runner: performs a crew run and prints summary + normalized JSON
	- Includes per-role suggestions built from top-ranked actors

---

## Environment Variables & Mapping

- Gemini: the app sets `GEMINI_API_KEY`; if only `GOOGLE_API_KEY` is present, it maps to `GEMINI_API_KEY` as well
- Serper: supports `SERPER_API_KEY` and `SERPERDEV_API_KEY`
- OMDb: `OMDB_API_KEY`
- Optional: `USD_TO_INR_RATE`

The Streamlit app loads `.env` at startup and standardizes these env vars for the SDK/tools.

---

## Troubleshooting

- Gemini INVALID_ARGUMENT / auth errors:
	- Ensure `.env` keys exist and are loaded
	- Use `GEMINI_API_KEY` or `GOOGLE_API_KEY`

- Empty crew output in UI:
	- The app normalizes output via several strategies; check the “Top-level keys” caption
	- Verify API keys and internet access; rerun

- Missing posters or sparse movies:
	- Confirm OMDb responses include `Poster` URLs and valid `imdbID`
	- Increase `n_similar` or adjust plot wording

---