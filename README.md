# Right Casting Choice AI

Three-agent casting workflow built on [crewAI](https://crewai.com) with:
- Character extraction (Gemini LLM)
- Similar movies via Serper, enriched by OMDb (INR conversions)
- Budget-aware actor ranking and per-role suggestions

Exclusively uses Gemini; no OpenAI calls. Environment variables are loaded from `.env`.

## Requirements
- Python `>=3.10,<3.14`
- Dependencies via `pyproject.toml` (includes `crewai[tools]`, `requests`, `streamlit`, `pandas`, `numpy`, `python-dotenv`, `google-generativeai`)

## Setup
1) Create a `.env` in the project root with:
```
GEMINI_API_KEY=your_gemini_key   # or GOOGLE_API_KEY
SERPER_API_KEY=your_serper_key   # or SERPERDEV_API_KEY
OMDB_API_KEY=your_omdb_key
USD_TO_INR_RATE=83.0             # optional override
```
2) Install dependencies:
```powershell
pip install -e .
```

## Run (Streamlit UI)
```powershell
streamlit run app.py
```
- Enter the plot, choose `Industry` (Hollywood or Bollywood), set `n_similar`, budget (in crores), and USD→INR.
- The UI shows:
	- Raw crew output (for debugging)
	- Posters from OMDb
	- Ranked actor candidates
	- Suggested actors per role (built from top-ranked names; if characters are unavailable, a default role is used)

## Run (CLI sample)
```powershell
python main.py
```
Outputs:
- Summary counts (characters, similar movies, candidates)
- Suggested actors per role
- Full JSON (normalized) result

## Industry behavior
- `Hollywood`: default, all actors/movies considered
- `Bollywood`: filters similar-movie candidates to Indian context using OMDb `_raw.Language` (Hindi) and `_raw.Country` (India)

## Project Structure
- `src/right_casting_choice_ai/crew.py`: defines the three agents and tasks, attaches Serper + OMDb tools, sets Gemini model.
- `src/right_casting_choice_ai/config/agents.yaml`: agent configs.
- `src/right_casting_choice_ai/config/tasks.yaml`: task prompts/outputs.
- `src/right_casting_choice_ai/tools/omdb.py`: OMDb tool with INR conversion.
- `app.py`: Streamlit UI orchestrating the crew output.
- `main.py`: CLI runner printing summary and normalized JSON.

## Troubleshooting
- Gemini auth errors (INVALID_ARGUMENT): ensure `.env` keys load; the app maps `GOOGLE_API_KEY` → `GEMINI_API_KEY` for SDK/LiteLLM.
- Empty crew output in UI: the app normalizes different `CrewOutput` shapes (`to_dict`, `raw`, `results`, `tasks_output`). Use the “Top-level keys” caption to guide parsing; rerun with valid keys.
- Posters missing: check OMDb responses include `Poster` URLs and keys are valid.

## License
This project is intended for learning and experimentation. No license is provided by default.
