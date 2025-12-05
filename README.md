# Right Casting Choice AI

An AI assistant that proposes casting choices for a film plot. It extracts character profiles, finds thematically similar movies, assembles actor candidates, estimates per‑film fees, and checks the total against a user budget. Built with CrewAI agents and a Streamlit UI.

## Highlights
- CrewAI workflow with 3 agents: character extractor, similar‑movies + OMDb aggregator, and budget‑aware ranker.
- Streamlit apps: interactive candidate tables, posters, and budget summaries.
- Robust JSON parsing and currency handling (USD→INR conversion for box office and fees).
- Optional per‑task visibility: prints each task’s output when available (`tasks_output`).

## Demo
[![Demo GIF](./demo.gif)](./demo.mp4)

## Repository Structure
```
right-casting-choice-ai/
├─ app1.py                # Streamlit app UI
├─ main.py                # CLI runner: run / train / replay / test
├─ Dockerfile             # Production container (serves app1.py on port 8000)
├─ pyproject.toml         # Package metadata
├─ requirements.txt       # Python dependencies
└─ src/right_casting_choice_ai/
   ├─ crew.py             # Crew definition (agents + tasks + crew)
   ├─ config/
   │  ├─ agents.yaml      # Agent prompts + behaviors
   │  └─ tasks.yaml       # Task descriptions + expected outputs
   └─ tools/
      ├─ omdb.py          # OMDb tool: normalizes data + converts to INR
      └─ custom_tool.py   # Example tool template
```

## Prerequisites
- Python 3.10–3.13
- API keys:
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for Gemini
  - `OMDB_API_KEY` for OMDb (movie metadata)
  - `SERPER_API_KEY` (or `SERPERDEV_API_KEY`) for Serper search

Optional:
- `USD_TO_INR_RATE` to override default 83.0 for conversions.

Example `.env`:
```
GEMINI_API_KEY=your_gemini_key
OMDB_API_KEY=your_omdb_key
SERPER_API_KEY=your_serper_key
USD_TO_INR_RATE=83.0
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

## Run the Streamlit App
Run the UI:
```powershell
streamlit run .\app1.py
```

During a run, the app:
- Extracts characters from your plot
- Finds similar movies and shows posters
- Builds an audition list with actor statistics and estimated fees
- Selects a top candidate per role and totals the cost against your budget
- Prints each task’s output if available (see “Task Outputs” section in the UI)

## CLI Usage
`main.py` provides a simple CLI for running or experimenting with the crew without the UI.

```powershell
# Run once with sample inputs, print summary + JSON
python .\main.py run

# Train the crew (toy training interface)
python .\main.py train 5 models/crew_state.json

# Replay a specific task by id
python .\main.py replay <task_id>

# Test with evaluator LLM id
python .\main.py test 3 gemini/gemini-2.5-flash
```

## Docker
The provided Dockerfile serves `app1.py` on port 8000.

```powershell
# Build
docker build -t right-casting-choice-ai .

# Run (Windows PowerShell)
docker run --rm -it -p 8000:8000 `
  -e GEMINI_API_KEY=$Env:GEMINI_API_KEY `
  -e OMDB_API_KEY=$Env:OMDB_API_KEY `
  -e SERPER_API_KEY=$Env:SERPER_API_KEY `
  right-casting-choice-ai
```
Open http://localhost:8000 in your browser.

## How It Works
- Crew: defined in `src/right_casting_choice_ai/crew.py`.
  - Agents: `character_extractor`, `similar_movies_and_omdb`, `budget_ranker`.
  - Tasks: `extract_characters_task`, `similar_movies_task`, `rank_candidates_task` configured via YAML.
  - Process: sequential (extract → movies → rank).
- Tools:
  - `omdb.py`: fetches OMDb details, normalizes poster/box office/budget/IMDB, converts USD→INR.
  - Serper (via `crewai_tools.SerperDevTool`): finds titles and salary/fee references.
- Streamlit apps:
  - Robust JSON scrapers for LLM outputs (code‑fenced/inline).
  - Currency parsing and conversions.
  - Poster grid + movie table (Title, Year, IMDB, BoxOffice; `app1.py` also shows optional Match Score).
  - Candidate tables with IMDB, average box office, fee, and notes.
  - Budget summary with industry‑aware units (Cr for Bollywood, M for Hollywood).
  - Task outputs (if the crew SDK exposes them as `result.tasks_output`): serialized and printed in “Task Outputs”.

The app prevents selecting the same actor for multiple roles and computes a grand total of the top pick per role.

## Configuration
- Industry selection influences candidate pools and currency units:
  - Bollywood → INR display in Crores (Cr), bias Indian/Hindi titles/actors
  - Hollywood → USD display in Millions (M), bias English titles/actors
- Exchange rate: configurable via UI and `USD_TO_INR_RATE`.
- Budget input: in M (Hollywood) or Cr (Bollywood), internally converted to raw integer units.

## Troubleshooting
- Rate limits/quota: UI shows a friendly error and stops; try again later. The code also respects retry hints when present.
- Empty candidates: Candidate availability depends on the crew output and available movie data.
- Missing posters: OMDb sometimes returns `N/A`; the UI skips those.
- Fees not showing: Ensure the ranker returns `implied_actor_fee_estimate` as a numeric INR integer and that sources are not low‑trust (ignored by `app1.py`).

## Development Notes
- The apps expect the crew to return a final JSON‑like result. When available, each `TaskOutput` is serialized and shown under “Task Outputs”.
- Update prompts in `src/right_casting_choice_ai/config/*.yaml` to tune behavior.
- Extend tools in `src/right_casting_choice_ai/tools/` for additional data sources.

## License
Licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.
